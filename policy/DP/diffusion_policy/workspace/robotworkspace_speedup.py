if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
# DP_ROOT points to .../policy/DP
DP_ROOT = pathlib.Path(__file__).resolve().parents[2]
from torch.utils.data import DataLoader
import copy

import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class AddIndicesWrapper:
    """
    Wrap a BaseImageDataset so that each returned batch carries the indices used to query it.
    DemoSpeedUp needs this to look up precomputed (entropy/label) arrays by dataset index.
    """
    def __init__(self, dataset: BaseImageDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # forward all other attributes / methods
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        batch["__indices"] = idx
        return batch

    def postprocess(self, batch, device):
        idx = batch.get("__indices", None)
        batch = self.dataset.postprocess(batch, device)
        if idx is not None:
            batch["__indices"] = idx
        return batch

    def get_validation_dataset(self):
        val_ds = self.dataset.get_validation_dataset()
        return AddIndicesWrapper(val_ds)


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # ========= DemoSpeedUp runtime caches ==========
        self._speedup_enabled = False
        self._speedup_apply_to_val = False
        self._speedup_labels_train = None  # np.ndarray [N, H] uint8
        self._speedup_labels_val = None    # np.ndarray [N, H] uint8
        self._speedup_low_v = 1
        self._speedup_high_v = 2

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # ========= Checkpoint dir policy (DP/checkpoints) ==========
        # 其它输出仍写到 hydra 的 output_dir；ckpt 统一写到 repo_root/DP/checkpoints 下
        ckpt_group_name = f"{cfg.task_name}_speedup"
        dp_ckpt_dir = os.path.join(str(DP_ROOT), "checkpoints", ckpt_group_name)
        os.makedirs(dp_ckpt_dir, exist_ok=True)
        dp_latest_ckpt_path = pathlib.Path(dp_ckpt_dir) / "latest.ckpt"

        # resume training
        if cfg.training.resume:
            # 1) 优先从 DP/checkpoints/<task_name>_speedup/latest.ckpt 续训
            if dp_latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {dp_latest_ckpt_path}")
                self.load_checkpoint(path=dp_latest_ckpt_path)
            else:
                # 2) 兼容旧版本：从 hydra output_dir/checkpoints/latest.ckpt 续训
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                    self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # ========= DemoSpeedUp switches ==========
        speedup_cfg = getattr(cfg, "speedup", None)
        do_label = False
        do_speedup = False
        if speedup_cfg is not None:
            do_label = bool(getattr(speedup_cfg, "label", False))
            do_speedup = bool(getattr(speedup_cfg, "enable", False) or getattr(speedup_cfg, "speedup", False))
            self._speedup_apply_to_val = bool(getattr(speedup_cfg, "apply_to_val", False))
            self._speedup_low_v = int(getattr(speedup_cfg, "low_v", 1))
            self._speedup_high_v = int(getattr(speedup_cfg, "high_v", 2))

        # Label mode: use a trained checkpoint to annotate entropy/labels, then exit.
        if do_label:
            label_ckpt_path = getattr(speedup_cfg, "label_ckpt_path", None) if speedup_cfg is not None else None
            if label_ckpt_path is not None:
                print(f"[DemoSpeedUp] Loading checkpoint for labeling: {label_ckpt_path}")
                self.load_checkpoint(path=pathlib.Path(label_ckpt_path))
            else:
                # 没显式指定 ckpt，则按顺序尝试：
                # 1) DP/checkpoints/<task_name>_speedup/latest.ckpt
                # 2) hydra output_dir/checkpoints/latest.ckpt (兼容旧版本)
                candidate_paths = [
                    dp_latest_ckpt_path,
                    self.get_checkpoint_path(tag="latest"),
                ]
                found_ckpt = None
                for p in candidate_paths:
                    if p is None:
                        continue
                    p = pathlib.Path(p)
                    if p.is_file():
                        found_ckpt = p
                        break
                if found_ckpt is None:
                    raise RuntimeError(
                        "[DemoSpeedUp] --label needs a trained checkpoint. "
                        "Can't find latest.ckpt in either DP/checkpoints or hydra output_dir. "
                        "Please set speedup.label_ckpt_path=/path/to/ckpt."
                    )
                print(f"[DemoSpeedUp] Loading checkpoint for labeling: {found_ckpt}")
                self.load_checkpoint(path=found_ckpt)

            label_device = torch.device(cfg.training.device)
            self.model.to(label_device)
            if self.ema_model is not None:
                self.ema_model.to(label_device)

            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()

            # label train split
            train_label_path = self._get_speedup_label_path(cfg.task.dataset.zarr_path, split="train")
            self._label_entropy_to_file(
                dataset=dataset,
                policy=policy,
                device=label_device,
                out_path=train_label_path,
                speedup_cfg=speedup_cfg,
            )
            # label val split (optional, but useful for debugging)
            val_dataset_for_label = dataset.get_validation_dataset()
            val_label_path = self._get_speedup_label_path(cfg.task.dataset.zarr_path, split="val")
            self._label_entropy_to_file(
                dataset=val_dataset_for_label,
                policy=policy,
                device=label_device,
                out_path=val_label_path,
                speedup_cfg=speedup_cfg,
            )
            print("[DemoSpeedUp] Labeling done. Exit now.")
            return

        # Speedup training: wrap dataset to keep indices, load precomputed labels.
        if do_speedup:
            self._speedup_enabled = True
            dataset = AddIndicesWrapper(dataset)

        train_dataloader = create_dataloader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        if self._speedup_enabled:
            train_label_path = self._get_speedup_label_path(cfg.task.dataset.zarr_path, split="train")
            self._speedup_labels_train = self._load_speedup_labels(train_label_path, expected_len=len(dataset))
            if self._speedup_apply_to_val:
                val_label_path = self._get_speedup_label_path(cfg.task.dataset.zarr_path, split="val")
                self._speedup_labels_val = self._load_speedup_labels(val_label_path, expected_len=len(val_dataset))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
            cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint (TopK + latest) — ckpt 统一写到 DP/checkpoints
        topk_manager = TopKCheckpointManager(save_dir=dp_ckpt_dir, **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        # DemoSpeedUp: compress low-entropy segments (train split)
                        if self._speedup_enabled:
                            batch = self._apply_speedup_to_batch(batch, device=device, split="train")
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (self.global_step % cfg.training.gradient_accumulate_every == 0):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps
                                is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = val_dataset.postprocess(batch, device)
                                # DemoSpeedUp (optional): apply to val split too
                                if self._speedup_enabled and self._speedup_apply_to_val:
                                    batch = self._apply_speedup_to_batch(batch, device=device, split="val")
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint (DP/checkpoints)
                is_ckpt_epoch = ((self.epoch + 1) % cfg.training.checkpoint_every) == 0
                is_last_epoch = (local_epoch_idx == (cfg.training.num_epochs - 1))
                if is_ckpt_epoch or is_last_epoch:
                    # 1) save latest.ckpt for resume
                    if getattr(cfg.checkpoint, "save_last_ckpt", True):
                        latest_path = os.path.join(dp_ckpt_dir, "latest.ckpt")
                        # BaseWorkspace.save_checkpoint(path=...) 不会自动拼 output_dir，因此这里必须给绝对路径
                        self.save_checkpoint(path=latest_path, use_thread=False)

                    # 2) TopK checkpoints (based on cfg.checkpoint.topk.monitor_key)
                    metric_dict = dict()
                    for key, value in step_log.items():
                        metric_dict[key.replace("/", "_")] = value
                    # filename 的 epoch 用 1-based 更直观
                    metric_dict["epoch"] = self.epoch + 1

                    monitor_key = cfg.checkpoint.topk.monitor_key
                    # 没有 rollout 时，默认用 -val_loss 充当 test_mean_score（等价于最小化 val_loss）
                    if monitor_key not in metric_dict:
                        if "val_loss" in metric_dict:
                            metric_dict[monitor_key] = -float(metric_dict["val_loss"])
                        else:
                            metric_dict[monitor_key] = -float(metric_dict["train_loss"])

                    topk_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path, use_thread=False)

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


    # ===================== DemoSpeedUp: entropy labeling =====================
    def _get_speedup_label_path(self, zarr_path: str, split: str = "train") -> str:
        """
        Sidecar file path for storing entropy/labels.
        We intentionally DO NOT modify the zarr itself to keep compatibility.
        """
        zarr_path = zarr_path.rstrip("/")
        return f"{zarr_path}.demospeedup_{split}.npz"

    def _load_speedup_labels(self, path: str, expected_len: int):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[DemoSpeedUp] label file not found: {path}. "
                f"Please run once with --label to generate it."
            )
        obj = np.load(path, allow_pickle=True)
        labels = obj["labels"]
        if labels.shape[0] != expected_len:
            raise RuntimeError(
                f"[DemoSpeedUp] label length mismatch: labels.shape[0]={labels.shape[0]} vs expected_len={expected_len}. "
                "Make sure you use the SAME dataset config (horizon/pad_before/pad_after/val_ratio/etc) when labeling."
            )
        return labels

    @torch.no_grad()
    def _compute_entropy(self, policy, obs_dict, num_samples: int = 10):
        """
        Estimate action entropy by sampling the policy multiple times.
        entropy[t] = mean(std(action_samples[:, t, :])) over action dims
        Returns: (B, H) torch.Tensor
        """
        action_samples = []
        for _ in range(num_samples):
            result = policy.predict_action(obs_dict)
            action_samples.append(result["action_pred"].detach())
        # [S, B, H, D]
        samples = torch.stack(action_samples, dim=0)
        entropy = torch.std(samples, dim=0).mean(dim=-1)
        return entropy

    @torch.no_grad()
    def _label_entropy_to_file(self, dataset, policy, device, out_path: str, speedup_cfg=None):
        """
        Iterate through the dataset by index (no DataLoader),
        compute entropy + binary labels, save as npz.
        """
        num_samples = int(getattr(speedup_cfg, "num_entropy_samples", 10)) if speedup_cfg is not None else 10
        entropy_quantile = float(getattr(speedup_cfg, "entropy_quantile", 0.3)) if speedup_cfg is not None else 0.3
        batch_size = int(getattr(speedup_cfg, "label_batch_size", self.cfg.dataloader.batch_size)) if speedup_cfg is not None else int(self.cfg.dataloader.batch_size)

        N = len(dataset)
        if N <= 0:
            raise RuntimeError("[DemoSpeedUp] dataset is empty, nothing to label.")

        # probe horizon H by running once
        probe_idx = np.arange(min(batch_size, N))
        if len(probe_idx) < batch_size:
            # wrap to keep batch-size consistent (dataset expects fixed batch)
            probe_idx = np.pad(probe_idx, (0, batch_size - len(probe_idx)), mode="wrap")
        probe_batch = dataset[probe_idx]
        probe_batch = dataset.postprocess(probe_batch, device)
        probe_obs = probe_batch["obs"]
        probe_action = policy.predict_action(probe_obs)["action_pred"]
        # action shape expected: [B, H, D]
        if probe_action.dim() != 3:
            raise RuntimeError(f"[DemoSpeedUp] unexpected action_pred shape: {tuple(probe_action.shape)}")
        H = int(probe_action.shape[1])

        entropy_all = np.zeros((N, H), dtype=np.float32)

        # main loop
        for start in tqdm.tqdm(range(0, N, batch_size), desc=f"[DemoSpeedUp] labeling {os.path.basename(out_path)}"):
            idx = np.arange(start, min(start + batch_size, N))
            if len(idx) < batch_size:
                # wrap to keep batch-size consistent
                idx_query = np.pad(idx, (0, batch_size - len(idx)), mode="wrap")
            else:
                idx_query = idx

            batch = dataset[idx_query]
            batch = dataset.postprocess(batch, device)
            obs_dict = batch["obs"]
            entropy = self._compute_entropy(policy, obs_dict, num_samples=num_samples)  # (B,H)
            entropy_np = entropy[: len(idx)].detach().cpu().numpy().astype(np.float32)
            entropy_all[idx] = entropy_np

        thr = float(np.quantile(entropy_all.reshape(-1), entropy_quantile))
        labels = (entropy_all < thr).astype(np.uint8)  # 1=low entropy -> can skip

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez(
            out_path,
            entropy=entropy_all,
            labels=labels,
            threshold=thr,
            entropy_quantile=entropy_quantile,
            num_entropy_samples=num_samples,
            low_v=int(self._speedup_low_v),
            high_v=int(self._speedup_high_v),
        )
        print(f"[DemoSpeedUp] Saved labels: {out_path} (N={N}, H={H}, thr={thr:.6f})")

    # ===================== DemoSpeedUp: training speedup =====================
    def _apply_speedup_to_batch(self, batch, device, split: str = "train"):
        """
        Use precomputed labels to compress action sequences in the batch.
        """
        indices = batch.get("__indices", None)
        if indices is None:
            # no index info, do nothing
            return batch

        # indices could be np.ndarray or torch.Tensor
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices)

        if split == "train":
            labels_arr = self._speedup_labels_train
        else:
            labels_arr = self._speedup_labels_val

        if labels_arr is None:
            return batch

        labels = torch.from_numpy(labels_arr[indices]).to(device=device)  # (B,H)
        actions = batch["action"]
        # sanity check: label horizon must match action horizon
        if (labels.ndim != 2) or (actions.ndim != 3) or (labels.shape[1] != actions.shape[1]):
            raise RuntimeError(
                f"[DemoSpeedUp] label/action horizon mismatch: labels.shape={tuple(labels.shape)}, "
                f"actions.shape={tuple(actions.shape)}. "
                "Please regenerate speedup labels with the SAME horizon/n_action_steps as training."
            )
        batch["action"] = self._speedup_actions(
            actions,
            labels,
            low_v=self._speedup_low_v,
            high_v=self._speedup_high_v,
        )

        return batch

    @staticmethod
    def _speedup_actions(actions: torch.Tensor, labels: torch.Tensor, low_v: int = 1, high_v: int = 2):
        """
        actions: (B,H,D)
        labels:  (B,H) uint8/bool/float, where 1 means low entropy (can skip more)
        Return new_actions with SAME shape as actions.
        Strategy:
          - pick indices by stepping low_v (normal) or high_v (skip) depending on label
          - put selected actions at the beginning
          - pad the rest by repeating the last selected action
        """
        assert actions.dim() == 3, f"actions must be (B,H,D), got {tuple(actions.shape)}"
        B, H, D = actions.shape
        new_actions = torch.zeros_like(actions)

        for b in range(B):
            idxs = RobotWorkspace._compute_speedup_indices(labels[b], low_v=low_v, high_v=high_v)
            k = len(idxs)
            new_actions[b, :k] = actions[b, idxs]
            if k > 0 and k < H:
                new_actions[b, k:] = new_actions[b, k - 1].unsqueeze(0).expand(H - k, D)
        return new_actions

    @staticmethod
    def _compute_speedup_indices(label_1d: torch.Tensor, low_v: int = 1, high_v: int = 2):
        """
        Compute indices to keep within a horizon based on binary labels.
        label_1d: (H,) tensor. 1=low entropy (can skip), 0=high entropy (keep dense)
        """
        H = int(label_1d.shape[0])
        idxs = [0]
        i = 0
        while i < H - 1:
            cur = int(label_1d[i].item())
            step = low_v
            if cur == 1 and (i + high_v) < H:
                window = label_1d[i : i + high_v]
                if torch.all(window == 1):
                    step = high_v
            i = i + step
            if i < H:
                idxs.append(int(i))
            else:
                break
        # ensure last index exists
        if idxs[-1] != H - 1:
            idxs.append(H - 1)
        # unique & sorted
        idxs = sorted(set(idxs))
        return idxs


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()