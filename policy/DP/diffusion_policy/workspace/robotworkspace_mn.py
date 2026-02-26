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
from methods.weight_methods import NashMTL


OmegaConf.register_new_resolver("eval", eval, replace=True)


# class RobotWorkspace(BaseWorkspace):
#     include_keys = ["global_step", "epoch"]
#
#     def __init__(self, cfg: OmegaConf, output_dir=None):
#         super().__init__(cfg, output_dir=output_dir)
#
#         # set seed
#         seed = cfg.training.seed
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)
#
#         # configure model
#         self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
#
#         self.ema_model: DiffusionUnetImagePolicy = None
#         if cfg.training.use_ema:
#             self.ema_model = copy.deepcopy(self.model)
#
#         # configure training state
#         self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
#
#         # configure training state
#         self.global_step = 0
#         self.epoch = 0
#
#     def run(self):
#         cfg = copy.deepcopy(self.cfg)
#         seed = cfg.training.seed
#         head_camera_type = cfg.head_camera_type
#
#         # resume training
#         if cfg.training.resume:
#             lastest_ckpt_path = self.get_checkpoint_path()
#             if lastest_ckpt_path.is_file():
#                 print(f"Resuming from checkpoint {lastest_ckpt_path}")
#                 self.load_checkpoint(path=lastest_ckpt_path)
#
#         # configure dataset
#         dataset: BaseImageDataset
#         dataset = hydra.utils.instantiate(cfg.task.dataset)
#         assert isinstance(dataset, BaseImageDataset)
#         train_dataloader = create_dataloader(dataset, **cfg.dataloader)
#         normalizer = dataset.get_normalizer()
#
#         # configure validation dataset
#         val_dataset = dataset.get_validation_dataset()
#         val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)
#
#         self.model.set_normalizer(normalizer)
#         if cfg.training.use_ema:
#             self.ema_model.set_normalizer(normalizer)
#
#         # configure lr scheduler
#         lr_scheduler = get_scheduler(
#             cfg.training.lr_scheduler,
#             optimizer=self.optimizer,
#             num_warmup_steps=cfg.training.lr_warmup_steps,
#             num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) //
#             cfg.training.gradient_accumulate_every,
#             # pytorch assumes stepping LRScheduler every epoch
#             # however huggingface diffusers steps it every batch
#             last_epoch=self.global_step - 1,
#         )
#
#         # configure ema
#         ema: EMAModel = None
#         if cfg.training.use_ema:
#             ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
#
#         # configure env
#         # env_runner: BaseImageRunner
#         # env_runner = hydra.utils.instantiate(
#         #     cfg.task.env_runner,
#         #     output_dir=self.output_dir)
#         # assert isinstance(env_runner, BaseImageRunner)
#         env_runner = None
#
#         # configure logging
#         # wandb_run = wandb.init(
#         #     dir=str(self.output_dir),
#         #     config=OmegaConf.to_container(cfg, resolve=True),
#         #     **cfg.logging
#         # )
#         # wandb.config.update(
#         #     {
#         #         "output_dir": self.output_dir,
#         #     }
#         # )
#
#         # configure checkpoint
#         topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
#                                              **cfg.checkpoint.topk)
#
#         # device transfer
#         device = torch.device(cfg.training.device)
#         self.model.to(device)
#         if self.ema_model is not None:
#             self.ema_model.to(device)
#         optimizer_to(self.optimizer, device)
#
#         # save batch for sampling
#         train_sampling_batch = None
#
#         if cfg.training.debug:
#             cfg.training.num_epochs = 2
#             cfg.training.max_train_steps = 3
#             cfg.training.max_val_steps = 3
#             cfg.training.rollout_every = 1
#             cfg.training.checkpoint_every = 1
#             cfg.training.val_every = 1
#             cfg.training.sample_every = 1
#
#         # training loop
#         log_path = os.path.join(self.output_dir, "logs.json.txt")
#
#         with JsonLogger(log_path) as json_logger:
#             for local_epoch_idx in range(cfg.training.num_epochs):
#                 step_log = dict()
#                 # ========= train for this epoch ==========
#                 if cfg.training.freeze_encoder:
#                     self.model.obs_encoder.eval()
#                     self.model.obs_encoder.requires_grad_(False)
#
#                 train_losses = list()
#                 with tqdm.tqdm(
#                         train_dataloader,
#                         desc=f"Training epoch {self.epoch}",
#                         leave=False,
#                         mininterval=cfg.training.tqdm_interval_sec,
#                 ) as tepoch:
#                     for batch_idx, batch in enumerate(tepoch):
#                         batch = dataset.postprocess(batch, device)
#                         if train_sampling_batch is None:
#                             train_sampling_batch = batch
#                         # compute loss
#                         raw_loss = self.model.compute_loss(batch)
#                         loss = raw_loss / cfg.training.gradient_accumulate_every
#                         loss.backward()
#
#                         # step optimizer
#                         if (self.global_step % cfg.training.gradient_accumulate_every == 0):
#                             self.optimizer.step()
#                             self.optimizer.zero_grad()
#                             lr_scheduler.step()
#
#                         # update ema
#                         if cfg.training.use_ema:
#                             ema.step(self.model)
#
#                         # logging
#                         raw_loss_cpu = raw_loss.item()
#                         tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
#                         train_losses.append(raw_loss_cpu)
#                         step_log = {
#                             "train_loss": raw_loss_cpu,
#                             "global_step": self.global_step,
#                             "epoch": self.epoch,
#                             "lr": lr_scheduler.get_last_lr()[0],
#                         }
#
#                         is_last_batch = batch_idx == (len(train_dataloader) - 1)
#                         if not is_last_batch:
#                             # log of last step is combined with validation and rollout
#                             json_logger.log(step_log)
#                             self.global_step += 1
#
#                         if (cfg.training.max_train_steps
#                                 is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
#                             break
#
#                 # at the end of each epoch
#                 # replace train_loss with epoch average
#                 train_loss = np.mean(train_losses)
#                 step_log["train_loss"] = train_loss
#
#                 # ========= eval for this epoch ==========
#                 policy = self.model
#                 if cfg.training.use_ema:
#                     policy = self.ema_model
#                 policy.eval()
#
#                 # run rollout
#                 # if (self.epoch % cfg.training.rollout_every) == 0:
#                 #     runner_log = env_runner.run(policy)
#                 #     # log all
#                 #     step_log.update(runner_log)
#
#                 # run validation
#                 if (self.epoch % cfg.training.val_every) == 0:
#                     with torch.no_grad():
#                         val_losses = list()
#                         with tqdm.tqdm(
#                                 val_dataloader,
#                                 desc=f"Validation epoch {self.epoch}",
#                                 leave=False,
#                                 mininterval=cfg.training.tqdm_interval_sec,
#                         ) as tepoch:
#                             for batch_idx, batch in enumerate(tepoch):
#                                 batch = dataset.postprocess(batch, device)
#                                 loss = self.model.compute_loss(batch)
#                                 val_losses.append(loss)
#                                 if (cfg.training.max_val_steps
#                                         is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
#                                     break
#                         if len(val_losses) > 0:
#                             val_loss = torch.mean(torch.tensor(val_losses)).item()
#                             # log epoch average validation loss
#                             step_log["val_loss"] = val_loss
#
#                 # run diffusion sampling on a training batch
#                 if (self.epoch % cfg.training.sample_every) == 0:
#                     with torch.no_grad():
#                         # sample trajectory from training set, and evaluate difference
#                         batch = train_sampling_batch
#                         obs_dict = batch["obs"]
#                         gt_action = batch["action"]
#
#                         result = policy.predict_action(obs_dict)
#                         pred_action = result["action_pred"]
#                         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
#                         step_log["train_action_mse_error"] = mse.item()
#                         del batch
#                         del obs_dict
#                         del gt_action
#                         del result
#                         del pred_action
#                         del mse
#
#                 # checkpoint
#                 if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
#                     # checkpointing
#                     save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
#                     self.save_checkpoint(f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt")  # TODO
#
#                 # ========= eval end for this epoch ==========
#                 policy.train()
#
#                 # end of epoch
#                 # log of last step is combined with validation and rollout
#                 json_logger.log(step_log)
#                 self.global_step += 1
#                 self.epoch += 1

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
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=self.model.parameters(),
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # whether to use Nash multi-task loss balancing
        # （你可以在 config.training.use_nash 里开关）
        self.use_nash = bool(getattr(cfg.training, "use_nash", False))

        # NashMTL 权重器（2 个任务：原始数据 + 下采样数据）
        if self.use_nash:
            device = torch.device(cfg.training.device)
            self.nashmtl = NashMTL(
                n_tasks=2,
                device=device,
                max_norm=getattr(cfg.training, "nash_max_norm", 1.0),
                update_weights_every=getattr(
                    cfg.training, "nash_update_weights_every", 50
                ),
                optim_niter=getattr(cfg.training, "nash_optim_niter", 10),
            )
        else:
            self.nashmtl = None

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # ========= 配置双数据集 =========
        # 原始数据集
        dataset_main: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        # 下采样数据集（需要在 config 里新增 cfg.task.dataset_downsample）
        dataset_downsample: BaseImageDataset = hydra.utils.instantiate(
            cfg.task.dataset_downsample
        )
        assert isinstance(dataset_main, BaseImageDataset)
        assert isinstance(dataset_downsample, BaseImageDataset)

        print(f"[RobotWorkspace] Main dataset zarr_path (cfg): {cfg.task.dataset.zarr_path}")
        print(f"[RobotWorkspace] Downsample dataset zarr_path (cfg): {cfg.task.dataset_downsample.zarr_path}")
        
        # 两个 dataloader（目前默认用同一个 dataloader 配置）
        train_dataloader_main = create_dataloader(dataset_main, **cfg.dataloader)
        train_dataloader_downsample = create_dataloader(
            dataset_downsample, **cfg.dataloader
        )

        # normalizer 用原始数据集的
        normalizer = dataset_main.get_normalizer()

        # validation 先只用原始数据集
        val_dataset = dataset_main.get_validation_dataset()
        val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # 每个 epoch 的 paired batch 数：以较长的 dataloader 为准，短的循环用
        len_main = len(train_dataloader_main)
        len_down = len(train_dataloader_downsample)
        num_train_batches_per_epoch = max(len_main, len_down)

        # ---- 计算“实际会发生多少次 optimizer.step()” ----
        steps_per_epoch_micro = num_train_batches_per_epoch
        if cfg.training.max_train_steps is not None:
            steps_per_epoch_micro = min(steps_per_epoch_micro, cfg.training.max_train_steps)

        accum = int(cfg.training.gradient_accumulate_every)
        steps_per_epoch_optim = steps_per_epoch_micro // accum
        if (steps_per_epoch_micro % accum) != 0:
            # 你启用了 flush，所以余数会额外产生一次 optimizer.step()
            steps_per_epoch_optim += 1

        total_optim_steps = steps_per_epoch_optim * int(cfg.training.num_epochs)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_optim_steps,
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # env_runner 暂时不用
        env_runner = None

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

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

                train_losses = []
                train_losses_main = []
                train_losses_downsample = []
                accum_step = 0

                # 让短的 dataloader 安全循环（不会缓存 batch）
                iter_main = infinite_dataloader(train_dataloader_main)
                iter_down = infinite_dataloader(train_dataloader_downsample)

                with tqdm.tqdm(
                        range(num_train_batches_per_epoch),
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        total=num_train_batches_per_epoch,
                ) as tepoch:
                    for batch_idx in tepoch:
                        batch_main = next(iter_main)
                        batch_downsample = next(iter_down)

                        batch_main = dataset_main.postprocess(batch_main, device)
                        batch_downsample = dataset_downsample.postprocess(batch_downsample, device)

                        if train_sampling_batch is None:
                            train_sampling_batch = batch_main

                        # ===== 双数据一致性自检：每个 epoch 的第 1 个 step 检查一次 =====
                        if batch_idx == 0:
                            sanity_check_dual_batches(batch_main, batch_downsample, allow_image_hw_mismatch=True)

                        # 两个任务各算一次 loss
                        raw_loss_main = self.model.compute_loss(batch_main)
                        raw_loss_downsample = self.model.compute_loss(
                            batch_downsample
                        )

                        if self.use_nash:
                            # 1）把两个 loss 堆成一个向量，并做梯度累积缩放
                            losses_vec = torch.stack(
                                (raw_loss_main, raw_loss_downsample)
                            ) / cfg.training.gradient_accumulate_every

                            # 1.5）只保留需要梯度的参数，避免 autograd 报错
                            shared_params = [
                                p for p in self.model.parameters() if p.requires_grad
                            ]

                            # 2）调用 NashMTL.backward：
                            #    - 内部会根据梯度求权重
                            #    - 会对 combined_loss 调用 backward()
                            #    - 会做梯度裁剪
                            combined_loss_scaled, extra_outputs = self.nashmtl.backward(
                                losses=losses_vec,
                                shared_parameters=shared_params,
                            )

                            # 3）从 extra_outputs 里拿到权重（仅用于日志和可视化）
                            weights = extra_outputs.get("weights", None)
                            if weights is not None:
                                w_main = float(weights[0].detach().cpu().item())
                                w_downsample = float(weights[1].detach().cpu().item())
                                # 用未缩放的原始 loss 重新算一个“真实尺度”的加权 loss 方便打印
                                raw_loss = (
                                    w_main * raw_loss_main
                                    + w_downsample * raw_loss_downsample
                                )
                            else:
                                # 理论上不会走到这里，保险起见
                                w_main = 0.5
                                w_downsample = 0.5
                                raw_loss = 0.5 * (
                                    raw_loss_main + raw_loss_downsample
                                )

                        else:
                            # 不开 Nash 就简单平均
                            raw_loss = 0.5 * (raw_loss_main + raw_loss_downsample)
                            w_main = 0.5
                            w_downsample = 0.5

                            # 这里仍然自己做 backward + grad accumulate
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                        accum_step += 1

                        # step optimizer: use accum_step (true micro-batch counter)
                        if (accum_step % cfg.training.gradient_accumulate_every) == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.global_step += 1  # global_step 统计“参数更新次数”

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging（步级）
                        raw_loss_cpu = raw_loss.item()
                        train_losses.append(raw_loss_cpu)
                        train_losses_main.append(raw_loss_main.item())
                        train_losses_downsample.append(raw_loss_downsample.item())

                        tepoch.set_postfix(
                            loss=raw_loss_cpu,
                            loss_main=raw_loss_main.item(),
                            loss_downsample=raw_loss_downsample.item(),
                            w_main=w_main,
                            w_downsample=w_downsample,
                            refresh=False,
                        )
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "train_loss_main": raw_loss_main.item(),
                            "train_loss_downsample": raw_loss_downsample.item(),
                            "nash_w_main": w_main,
                            "nash_w_downsample": w_downsample,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "B_main": int(batch_main["action"].shape[0]),
                            "B_down": int(batch_downsample["action"].shape[0]),
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (num_train_batches_per_epoch - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            json_logger.log(step_log)

                        if (
                            cfg.training.max_train_steps is not None
                            and batch_idx >= (cfg.training.max_train_steps - 1)
                        ):
                            break

                # flush last partial accumulation (if any)
                if (accum_step % cfg.training.gradient_accumulate_every) != 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                    self.global_step += 1

                # ========= end of train epoch ==========
                # 用 epoch 平均代替最后一步
                train_loss = np.mean(train_losses) if len(train_losses) > 0 else 0.0
                train_loss_main = (
                    np.mean(train_losses_main) if len(train_losses_main) > 0 else 0.0
                )
                train_loss_downsample = (
                    np.mean(train_losses_downsample)
                    if len(train_losses_downsample) > 0
                    else 0.0
                )
                step_log["train_loss"] = train_loss
                step_log["train_loss_main"] = train_loss_main
                step_log["train_loss_downsample"] = train_loss_downsample

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation（只在原数据集上）
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset_main.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= (cfg.training.max_val_steps - 1)
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
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

                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing：名字包含两个 zarr
                    main_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    downsample_name = pathlib.Path(
                        self.cfg.task.dataset_downsample.zarr_path
                    ).stem

                    task_tag = cfg.task.name

                    save_name = f"{task_tag}"
                    self.save_checkpoint(
                        f"checkpoints/{save_name}-{seed}/{self.epoch + 1}.ckpt"
                    )

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                json_logger.log(step_log)
                self.epoch += 1



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


def infinite_dataloader(dataloader):
    """Safe cycle: re-create dataloader iterator when exhausted; does NOT cache batches."""
    while True:
        for batch in dataloader:
            yield batch


def sanity_check_dual_batches(batch_main, batch_down, *, allow_image_hw_mismatch=True):
    """
    检查 main/downsample 两个 batch 是否结构一致：
    - 必须包含 obs/action
    - action shape 必须一致（至少 batch 维一致）
    - obs 的 key 必须一致
    - 对 obs tensor：允许图像 H/W 不同（下采样），但 batch/time/channel 等维度必须一致
    """
    assert isinstance(batch_main, dict) and isinstance(batch_down, dict), "batch 必须是 dict"
    assert "obs" in batch_main and "action" in batch_main, "main batch 缺少 obs/action"
    assert "obs" in batch_down and "action" in batch_down, "down batch 缺少 obs/action"

    obs_m = batch_main["obs"]
    obs_d = batch_down["obs"]
    act_m = batch_main["action"]
    act_d = batch_down["action"]

    # action: 至少 shape 完全一致（最保险）
    assert tuple(act_m.shape) == tuple(act_d.shape), (
        f"action shape 不一致: main={tuple(act_m.shape)} down={tuple(act_d.shape)}"
    )

    # obs: key 必须一致
    assert isinstance(obs_m, dict) and isinstance(obs_d, dict), "obs 必须是 dict"
    assert set(obs_m.keys()) == set(obs_d.keys()), (
        f"obs keys 不一致: main={sorted(list(obs_m.keys()))} down={sorted(list(obs_d.keys()))}"
    )

    # obs tensor shape check
    for k in obs_m.keys():
        xm = obs_m[k]
        xd = obs_d[k]
        # 只对 tensor 做 shape 检查（有些 meta 可能不是 tensor）
        if torch.is_tensor(xm) and torch.is_tensor(xd):
            if tuple(xm.shape) == tuple(xd.shape):
                continue

            if not allow_image_hw_mismatch:
                raise AssertionError(f"obs[{k}] shape 不一致: main={tuple(xm.shape)} down={tuple(xd.shape)}")

            # 允许图像类 tensor 的 H/W 不同：
            # 常见 shape: [B,C,H,W] 或 [B,T,C,H,W]
            if xm.ndim == 4 and xd.ndim == 4:
                # B,C 必须一致，H/W 允许不同
                assert xm.shape[0] == xd.shape[0] and xm.shape[1] == xd.shape[1], \
                    f"obs[{k}] BC 不一致: main={tuple(xm.shape)} down={tuple(xd.shape)}"
            elif xm.ndim == 5 and xd.ndim == 5:
                # B,T,C 必须一致，H/W 允许不同
                assert xm.shape[0] == xd.shape[0] and xm.shape[1] == xd.shape[1] and xm.shape[2] == xd.shape[2], \
                    f"obs[{k}] BTC 不一致: main={tuple(xm.shape)} down={tuple(xd.shape)}"
            else:
                # 其它维度的 obs 不应该 mismatch
                raise AssertionError(f"obs[{k}] shape 不一致且不是可接受的图像下采样差异: main={tuple(xm.shape)} down={tuple(xd.shape)}")




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
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
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