import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import types
import torchvision.transforms as transforms

import hydra
from omegaconf import OmegaConf

# build_CNNMLP_model_and_optimizer 用来构建 diffusion_policy 需要的 encoder（与 DemoSpeedUp 一致）
try:
    from detr.main import build_CNNMLP_model_and_optimizer
except Exception:
    from .detr.main import build_CNNMLP_model_and_optimizer


class DP:
    """
    DemoSpeedUp 风格的 DP 推理封装：
    - 加载 diffusion_policy cfg + encoder
    - 加载 ckpt_dir/policy_last.ckpt + ckpt_dir/dataset_stats.pkl
    - 提供 get_action(obs) -> numpy (1, 14)，可被 deploy_policy.py 的 for action in actions 执行
    """

    def __init__(self, args_override: dict, RoboTwin_Config=None):
        self.device = torch.device(args_override.get("device", "cuda:0"))

        # 1) 读取 diffusion_policy 的 yaml
        cfg_path = (
            args_override.get("diffusion_policy_cfg")
            or args_override.get("demospeedup_policy_yaml")
            or args_override.get("dp_cfg")
        )
        if not cfg_path:
            raise ValueError(
                "DP 需要 diffusion_policy 配置文件路径。请在 deploy_policy.yml 或命令行 overrides 里提供 "
                "diffusion_policy_cfg (或 demospeedup_policy_yaml / dp_cfg)。"
            )
        self.cfg = OmegaConf.load(cfg_path)

        # ===== [新增] 强制用 diffusion_policy_cfg 里的 action_dim（保证和 ckpt 对齐）=====
        # 你的 image_aloha_diffusion_policy_cnn.yaml 里 action.shape = [14]
        yaml_action_dim = int(self.cfg.policy.shape_meta.action.shape[0])

        # 覆盖 args_override，确保后面 encoder / policy 都按 14 构建
        args_override["action_dim"] = yaml_action_dim
        args_override["state_dim"] = yaml_action_dim  # detr builder 用 state_dim

        print(f"[DP] yaml action_dim = {yaml_action_dim} (override action_dim/state_dim)")
        # ===== [新增结束] =====

        # 2) 构建 encoder（尽量复用你现有 ACT 的参数；也允许用户提供 dp_encoder 子配置）
        from argparse import Namespace

        # 2) 构建 encoder（尽量复用你现有 ACT 的参数；也允许用户提供 dp_encoder 子配置）
        from argparse import Namespace

        # ===== [新增] 强制让 encoder 至少有 1 个相机（否则会构建成 state-only，flattened_features 为空）=====
        cam_name = args_override.get("dp_camera_key", "cam_high")  # 离线用 cam_high；在线可用 --dp_camera_key 覆盖
        if isinstance(cam_name, str):
            cam_list = [cam_name]
        else:
            cam_list = list(cam_name)

        # 常见字段名都 setdefault 一遍（不覆盖用户显式设置）
        args_override.setdefault("camera_names", cam_list)  # 最常见
        args_override.setdefault("cam_names", cam_list)
        args_override.setdefault("camera_keys", cam_list)
        args_override.setdefault("cameras", cam_list)
        args_override.setdefault("num_cameras", len(cam_list))

        # 有些实现用字符串形式
        args_override.setdefault("camera_names_str", ",".join(cam_list))

        # 有些实现需要显式开 vision（setdefault 不会覆盖已有配置）
        args_override.setdefault("use_vision", True)
        args_override.setdefault("use_image", True)
        args_override.setdefault("use_rgb", True)
        # ===== [新增结束] =====

        # build_CNNMLP_model_and_optimizer 需要 Namespace；否则会去 parse sys.argv 然后报缺参数
        encoder_args_dict = dict(args_override)

        # build_CNNMLP_model_and_optimizer 需要 Namespace；否则会去 parse sys.argv 然后报缺参数
        encoder_args_dict = dict(args_override)

        # 补齐 DETR parser 要求的字段（否则它还是会报 required）
        encoder_args_dict.setdefault("ckpt_dir", encoder_args_dict.get("ckpt_dir", ""))
        encoder_args_dict.setdefault("policy_class", encoder_args_dict.get("policy_class", "DP"))
        encoder_args_dict.setdefault("task_name", encoder_args_dict.get("task_name", ""))
        encoder_args_dict.setdefault("seed", int(encoder_args_dict.get("seed", 0)))
        encoder_args_dict.setdefault("num_epochs", int(encoder_args_dict.get("num_epochs", 2000)))

        # 关键：DETR 那边要 state_dim；必须与 DP ckpt 的 action_dim 一致（由 yaml 决定）
        encoder_args_dict["state_dim"] = int(args_override["state_dim"])
        encoder_args_dict["action_dim"] = int(args_override["action_dim"])

        encoder_args = Namespace(**encoder_args_dict)
        encoder, _ = build_CNNMLP_model_and_optimizer(encoder_args)
        # ===== [新增] 防呆：encoder 必须真的有 backbones，否则离线一定会炸 =====
        if hasattr(encoder, "backbones"):
            try:
                n_b = len(encoder.backbones)
                print(f"[DP] encoder.backbones = {n_b}")
                if n_b == 0:
                    raise RuntimeError(
                        "[DP] encoder.backbones is empty (0 cameras). "
                        "Pass dp_camera_key/camera_names to build a vision encoder."
                    )
            except TypeError:
                pass
        # ===== [新增结束] =====

        # 3) 构建 diffusion policy
        # cfg.policy._target_ = diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy
        self.policy: nn.Module = hydra.utils.instantiate(self.cfg.policy, encoder=encoder)

        # ===== [新增] 强制把 obs_encoder 的 action head 改成 yaml_action_dim（14），避免 ckpt 14 vs 当前 16 =====
        self._force_obs_encoder_action_dim(yaml_action_dim)
        # ===== [新增结束] =====

        # 补 proj_head + get_encoder_feature（否则 predict_action 会报 CNNMLP 没这个方法）
        self._ensure_obs_encoder_feature_api()

        self.policy.to(self.device)
        self.policy.eval()
        # ===== [新增] 在线/离线可控开关（必须在 __init__ 内）=====
        self.dp_qpos_norm = DP._parse_bool(args_override.get("dp_qpos_norm", True), default=True)
        self.dp_action_denorm = DP._parse_bool(args_override.get("dp_action_denorm", True), default=True)

        self.dp_imagenet_norm = args_override.get("dp_imagenet_norm", "auto")
        if isinstance(self.dp_imagenet_norm, str):
            self.dp_imagenet_norm = self.dp_imagenet_norm.strip().lower()
        # ===== [新增结束] =====

        import torch.nn.functional as F  # 文件顶部加：与其他 import 同级

        # 4) 推理时用的相机：训练数据是 cam_high（HWC uint8）
        self.camera_key = args_override.get("dp_camera_key", "cam_high")
        # ===== debug 开关：在线采样保存 =====
        self.dp_debug = DP._parse_bool(args_override.get("dp_debug", False), default=False)
        self.dp_debug_dir = args_override.get("dp_debug_dir", "./dp_debug")
        self.dp_debug_every = int(args_override.get("dp_debug_every", 1))  # 每 N 步采样一次
        self._dp_debug_step = 0
        # ===== debug 开关结束 =====

        # 允许自动 fallback：如果 obs 里没有 cam_high，就按候选顺序找
        self.camera_candidates = args_override.get(
            "dp_camera_candidates",
            ["cam_high", "head_cam", "head_camera", "front_cam", "front_camera", "wrist_cam", "wrist_camera"]
        )

        # 从 cfg 里读期望的图像尺寸（如果 yaml 里 shape_meta 写的是 [3,H,W]）
        self.expected_image_hw = None
        try:
            shp = self.cfg.policy.shape_meta.obs.agentview_image.shape
            if isinstance(shp, (list, tuple)) and len(shp) == 3:
                self.expected_image_hw = (int(shp[1]), int(shp[2]))
        except Exception:
            pass

        self._printed_cam_once = False

        # 5) 设定 num_queries（DP 用 horizon；与 DemoSpeedUp 思路一致）
        self.num_queries = int(getattr(self.cfg.policy, "horizon", 24))
        self.temporal_agg = DP._parse_bool(args_override.get("temporal_agg", False), default=False)

        # 与 ACT 保持一致的 temporal_agg 缓存结构（必须与 yaml/ckpt 对齐）
        self.state_dim = int(args_override["action_dim"])
        self.max_timesteps = 3000
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            self.all_time_actions = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.num_queries, self.state_dim],
                device=self.device,
                dtype=torch.float32,
            )
            print(f"[DP] Temporal aggregation enabled with {self.num_queries} queries")

        self.t = 0
        self.all_actions = None

        # 6) 读取 stats + ckpt
        ckpt_dir = args_override.get("ckpt_dir", "")
        if not ckpt_dir:
            raise ValueError("[DP] args_override.ckpt_dir 为空：请指向包含 policy_last.ckpt 与 dataset_stats.pkl 的目录。")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[DP] 找不到 dataset_stats.pkl: {stats_path}")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        print(f"[DP] Loaded normalization stats from {stats_path}")

        ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[DP] 找不到 policy_last.ckpt: {ckpt_path}")

        self._load_ckpt(ckpt_path)

    @staticmethod
    def _parse_bool(x, default=False):
        if isinstance(x, bool):
            return x
        if x is None:
            return default
        if isinstance(x, (int, np.integer)):
            return bool(int(x))
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("1", "true", "t", "yes", "y", "on"):
                return True
            if s in ("0", "false", "f", "no", "n", "off", "none", "null", ""):
                return False
        # 兜底：别再用 bool("false") 这种坑
        return default

    @staticmethod
    def _normalize_state_dict_keys(sd: dict) -> dict:
        """
        把 checkpoint 的 key 规范化成当前 policy 期望的命名：
        - 去掉常见包装前缀：module. / model. / policy.
        - 处理 model.model.xxx 这种多嵌一层的情况
        """
        if not isinstance(sd, dict) or len(sd) == 0:
            return sd

        keys = list(sd.keys())
        for p in ["module.", "model.", "policy."]:
            cnt = sum(k.startswith(p) for k in keys)
            if cnt >= 0.8 * len(keys):
                sd = {k[len(p):]: v for k, v in sd.items()}
                keys = list(sd.keys())

        # 特判：model.model.xxx -> model.xxx
        sd = {
            (k.replace("model.model.", "model.", 1) if k.startswith("model.model.") else k): v
            for k, v in sd.items()
        }
        return sd

    @staticmethod
    def _load_policy_weights_clean(policy: torch.nn.Module, ckpt_path: str):
        """
        先 normalize key，然后优先 strict=True 干净加载。
        如果 strict=True 仍失败（比如 shape 不一致），再做 shape-filter + strict=False 兜底。
        """
        payload = torch.load(ckpt_path, map_location="cpu")
        sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

        sd = DP._normalize_state_dict_keys(sd)

        # 你原逻辑里手动 pop proj_head，这里保留（无害且能避免 shape mismatch）
        sd.pop("obs_encoder.proj_head.weight", None)
        sd.pop("obs_encoder.proj_head.bias", None)

        # 1) 优先 strict=True：正常情况下这里会直接成功，不再喷一大坨日志
        try:
            policy.load_state_dict(sd, strict=True)
            print(f"[DP] Loaded policy weights from {ckpt_path} (strict=True, normalized keys)")
            return
        except RuntimeError as e:
            print(f"[DP] strict=True failed after key-normalize, fallback to shape-filter + strict=False. err={e}")

        # 2) 兜底：过滤 shape 不一致的 key（否则 strict=False 也会报 shape mismatch）
        model_sd = policy.state_dict()
        filtered = {}
        skipped_shape = []

        for k, v in sd.items():
            if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped_shape.append(k)

        incompat = policy.load_state_dict(filtered, strict=False)
        print(f"[DP] Loaded policy weights from {ckpt_path} (strict=False, filtered)")
        print(
            f"[DP] missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)} skipped_shape={len(skipped_shape)}")

    def _ensure_obs_encoder_feature_api(self):
        """
        让 RoboTwin 的 CNNMLP 具备 diffusion_policy 需要的接口：
        - 补 proj_head（ckpt 里有）
        - 补 get_encoder_feature（DiffusionUnetHybridImagePolicy 会调用）
        输出维度必须是 768+14=782，和 diffusion_unet_hybrid_image_policy.py 里 hardcode 的 obs_feature_dim 对齐
        """
        oe = self.policy.obs_encoder

        # 1) 补 proj_head（ckpt 里有 obs_encoder.proj_head.*）
        if not hasattr(oe, "action_head") or not isinstance(oe.action_head, nn.Linear):
            raise RuntimeError("[DP] obs_encoder 没有 nn.Linear 的 action_head，无法用 hook 抓取 1000-d feature")

        # 注意：不要在这里强行创建 proj_head
        # DemoSpeedUp ckpt 里的 proj_head 是 (768 -> 64)，与你当前 CNNMLP 的 proj_head 定义不一致
        # 我们在 get_encoder_feature 里直接用 action_head 输入 feat1000 的前 768 维作为视觉特征

        # 2) 注入 get_encoder_feature（如果已经有就不覆盖）
        if hasattr(oe, "get_encoder_feature"):
            return

        def get_encoder_feature(self_encoder, nobs: dict):
            """
            nobs keys 必须是 yaml 里的：
              - agentview_image: (B, To, 3, H, W)
              - agent_pos:      (B, To, 14)
            返回: (B*To, 782) 或 (B, 782) 都行。
            这里 To=1（你 yaml n_obs_steps=1），我们返回 (B, 782)。
            """
            img = nobs["agentview_image"]
            pos = nobs["agent_pos"]

            # To=1，取第 0 帧
            if img.dim() == 5:
                img0 = img[:, 0]  # (B,3,H,W)
            else:
                img0 = img

            if pos.dim() == 3:
                pos0 = pos[:, 0]  # (B,14)
            else:
                pos0 = pos

            # CNNMLP 一般吃 (B, Ncam, 3, H, W)，我们只有单相机，补一维
            if img0.dim() == 4:
                img_in = img0.unsqueeze(1)  # (B,1,3,H,W)
            else:
                img_in = img0

            qpos_in = pos0  # (B,14)
            env_state = None

            captured = {}

            def hook(module, inputs, output):
                # Linear 的 inputs[0] 就是 (B, F) feature
                captured["feat"] = inputs[0]

            handle = None
            if hasattr(self_encoder, "action_head") and isinstance(self_encoder.action_head, nn.Linear):
                handle = self_encoder.action_head.register_forward_hook(hook)

            # 触发一次 forward（CNNMLP 的签名在 RoboTwin 常见是 (qpos, image, env_state)）
            try:
                out = self_encoder(qpos_in, img_in, env_state)
            except TypeError:
                out = self_encoder(img_in, qpos_in, env_state)

            if handle is not None:
                handle.remove()

            # ===== [新增] 如果 hook 没抓到 feature，就走 fallback =====
            if "feat" in captured:
                feat_any = captured["feat"]  # (B, F)
            else:
                # 1) 尝试从 encoder/backbone 直接取（不同实现命名不同）
                feat_any = None

                # 常见：self_encoder.backbone(img) 或 self_encoder.encoder(img)
                for attr in ["backbone", "encoder", "cnn", "vision_backbone"]:
                    if hasattr(self_encoder, attr):
                        mod = getattr(self_encoder, attr)
                        try:
                            tmp = mod(img_in)
                            # tmp 可能是 (B, C) 或 (B, C, 1, 1)，统一 flatten
                            if isinstance(tmp, (list, tuple)):
                                tmp = tmp[0]
                            tmp = tmp.flatten(1)
                            feat_any = tmp
                            break
                        except Exception:
                            pass

                # 2) 最后兜底：用 out（action）扩展成 feature（只为先跑通 eval）
                if feat_any is None:
                    tmp = out
                    if isinstance(tmp, (list, tuple, dict)):
                        # 尽量取出一个 tensor
                        if isinstance(tmp, dict):
                            tmp = next(iter(tmp.values()))
                        else:
                            tmp = tmp[0]
                    if not torch.is_tensor(tmp):
                        raise RuntimeError("[DP] Cannot derive encoder feature: hook failed and no backbone found.")
                    tmp = tmp.flatten(1)  # (B, Da)
                    # repeat/pad 到至少 768
                    repeat = (768 + tmp.shape[1] - 1) // tmp.shape[1]
                    feat_any = tmp.repeat(1, repeat)[:, :768].contiguous()  # (B,768)

            # ===== [新增结束] =====

            # 现在把 feat_any 变成 feat768
            feat_any = feat_any.flatten(1)
            if feat_any.shape[1] >= 768:
                feat768 = feat_any[:, :768].contiguous()
            else:
                # 不足 768 就 pad
                pad = 768 - feat_any.shape[1]
                feat768 = torch.cat([feat_any, feat_any.new_zeros(feat_any.shape[0], pad)], dim=1)

            feat782 = torch.cat([feat768, qpos_in], dim=-1)  # (B,782)
            return feat782

        oe.get_encoder_feature = types.MethodType(get_encoder_feature, oe)
        print("[DP] Patched obs_encoder.get_encoder_feature() for diffusion_policy")

    def _force_obs_encoder_action_dim(self, action_dim: int):
        """
        解决你现在遇到的核心问题：
        - ckpt 里 obs_encoder.action_head / obs_encoder.mlp.4 是 14 维
        - 但当前构建出来可能是 16 维
        这里直接把当前模型的输出层替换成 14 维，确保能 load ckpt。
        """
        if not hasattr(self.policy, "obs_encoder"):
            print("[DP] Warning: policy has no obs_encoder, skip resizing.")
            return

        oe = self.policy.obs_encoder

        # 1) action_head: Linear(in_features, out_features)
        if hasattr(oe, "action_head") and isinstance(oe.action_head, nn.Linear):
            if oe.action_head.out_features != action_dim:
                in_f = oe.action_head.in_features
                print(f"[DP] Resize obs_encoder.action_head: {oe.action_head.out_features} -> {action_dim}")
                oe.action_head = nn.Linear(in_f, action_dim)

        # 2) mlp 最后一层（你的报错是 obs_encoder.mlp.4）
        if hasattr(oe, "mlp"):
            # 常见是 nn.Sequential
            if isinstance(oe.mlp, nn.Sequential) and len(oe.mlp) > 0:
                last = oe.mlp[-1]
                if isinstance(last, nn.Linear) and last.out_features != action_dim:
                    in_f = last.in_features
                    print(f"[DP] Resize obs_encoder.mlp[-1]: {last.out_features} -> {action_dim}")
                    oe.mlp[-1] = nn.Linear(in_f, action_dim)
            else:
                # 如果不是 Sequential，但有索引 4（对应 mlp.4）
                try:
                    last = oe.mlp[4]
                    if isinstance(last, nn.Linear) and last.out_features != action_dim:
                        in_f = last.in_features
                        print(f"[DP] Resize obs_encoder.mlp[4]: {last.out_features} -> {action_dim}")
                        oe.mlp[4] = nn.Linear(in_f, action_dim)
                except Exception:
                    pass

    def _load_ckpt(self, ckpt_path: str):
        # 方案A：先规范化 key，再“干净加载”
        DP._load_policy_weights_clean(self.policy, ckpt_path)

    def pre_process(self, qpos: np.ndarray) -> np.ndarray:
        return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]

    def _pick_camera_key(self, obs: dict) -> str:
        # 1) 优先用用户指定的 camera_key
        if self.camera_key in obs:
            return self.camera_key
        # 2) 再按候选列表找
        for k in self.camera_candidates:
            if k in obs:
                return k
        # 3) 最后再做个模糊匹配（比如 keys 里包含 cam_high）
        for k in obs.keys():
            if "cam_high" in k:
                return k
        raise KeyError(
            f"[DP] 找不到相机图像。camera_key={self.camera_key}, candidates={self.camera_candidates}, obs_keys={list(obs.keys())}")

    def _to_chw_float01(self, img: np.ndarray) -> np.ndarray:
        """
        支持：
        - HWC uint8 (0~255)   -> CHW float32 (0~1)
        - CHW float32         -> CHW float32 (0~1)（若像素看起来像 0~255 会自动 /255）
        """
        img = np.asarray(img)

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.5:  # 很可能是 0~255 的 float
                img = img / 255.0

        # HWC -> CHW
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 3 and img.shape[0] == 3:
            pass
        else:
            raise ValueError(f"[DP] Unsupported image shape: {img.shape} (expect HWC or CHW with 3 channels)")

        return img

    def post_process(self, action: np.ndarray) -> np.ndarray:
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def _debug_dump(self, payload: dict):
        if not self.dp_debug:
            return
        if (self._dp_debug_step % self.dp_debug_every) != 0:
            self._dp_debug_step += 1
            return
        self._dp_debug_step += 1

        os.makedirs(self.dp_debug_dir, exist_ok=True)
        out = os.path.join(self.dp_debug_dir, f"step_{self.t:06d}.npz")

        # npz 不能直接存 python 字符串 dict，转成 object
        np.savez_compressed(out, payload=np.array(payload, dtype=object))

    def reset(self):
        """给 deploy_policy.reset_model 调用"""
        self.t = 0
        if self.temporal_agg:
            self.all_time_actions.zero_()
            print("[DP] Reset temporal aggregation state")

    @torch.no_grad()
    def get_action(self, obs: dict):
        """
        obs 来自 deploy_policy.encode_obs()
        返回 numpy shape (1, 14)，外层 deploy_policy.eval 会 for action in actions 执行一次 take_action。
        """
        # 1) qpos
        # 1) qpos（同时保留 raw 和 norm 统计）
        raw_qpos = np.array(obs["qpos"], dtype=np.float32)
        if self.dp_qpos_norm:
            qpos_used = self.pre_process(raw_qpos.copy())
        else:
            qpos_used = raw_qpos.copy()

        qpos_t = torch.from_numpy(qpos_used).float().to(self.device).unsqueeze(0).unsqueeze(1)

        # 2) image（记录 dtype/range + 是否已 ImageNet normalize）
        img_key = self._pick_camera_key(obs)
        img_np = np.asarray(obs[img_key])

        already_imagenet = False
        if img_np.dtype == np.uint8:
            img_np_f = img_np.astype(np.float32) / 255.0
        else:
            img_np_f = img_np.astype(np.float32)
            mn, mx = float(img_np_f.min()), float(img_np_f.max())
            if mn < -0.5:  # 含明显负值：高度怀疑已做过 normalize
                already_imagenet = True
            elif mx > 1.5:  # 无负值但 >1：大概率是 0~255 float
                img_np_f = img_np_f / 255.0
            else:
                pass  # 0~1 float

        # HWC -> CHW
        if img_np_f.ndim == 3 and img_np_f.shape[-1] == 3:
            img_chw = np.transpose(img_np_f, (2, 0, 1))
        elif img_np_f.ndim == 3 and img_np_f.shape[0] == 3:
            img_chw = img_np_f
        else:
            raise ValueError(f"[DP] Unsupported image shape: {img_np_f.shape}")

        img_t = torch.from_numpy(img_chw).to(self.device).unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)

        # resize 到 cfg 期望尺寸（如果你已修好 expected_image_hw 解析）
        if self.expected_image_hw is not None and tuple(img_t.shape[-2:]) != tuple(self.expected_image_hw):
            x = img_t.reshape(-1, 3, img_t.shape[-2], img_t.shape[-1])
            x = torch.nn.functional.interpolate(x, size=self.expected_image_hw, mode="bilinear", align_corners=False)
            img_t = x.reshape(1, 1, 3, self.expected_image_hw[0], self.expected_image_hw[1])

        # 只有没做过 imagenet normalize 才做
        do_norm = True
        if self.dp_imagenet_norm == "auto":
            do_norm = (not already_imagenet)
        else:
            do_norm = DP._parse_bool(self.dp_imagenet_norm, default=True)

        if do_norm:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            img_t = normalize(img_t)

        if not self._printed_cam_once:
            img_like = []
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    img_like.append((k, tuple(v.shape), str(v.dtype)))
            print("[DP] image-like keys:", img_like)

            mn0, mx0 = float(img_np_f.min()), float(img_np_f.max())
            print(f"[DP] Using camera='{img_key}', img_dtype={img_np.dtype}, img_range=({mn0:.3f},{mx0:.3f}), "
                  f"already_imagenet={already_imagenet}, tensor={tuple(img_t.shape)}, expected_hw={self.expected_image_hw}")
            self._printed_cam_once = True

        # 3) 只在 query_frequency 时刻重新 query 一段动作序列（仿 DemoSpeedUp）
        if self.t % self.query_frequency == 0:
            # diffusion_policy 的 shape_meta 里 obs keys 是 agentview_image / agent_pos
            obs_dict = {
                "agent_pos": qpos_t,  # (1,1,14)
                "agentview_image": img_t,  # (1,1,3,H,W) 你已经是这个形状
            }
            result = self.policy.predict_action(obs_dict)
            self.all_actions = result["action_pred"]  # (B, horizon, Da)

        # 4) 取当前步 action（以及可选 temporal_agg）
        if self.temporal_agg:
            self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = self.all_actions
            actions_for_curr_step = self.all_time_actions[:, self.t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]

            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(1)

            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)  # (1,Da)
        else:
            raw_action = self.all_actions[:, self.t % self.query_frequency]  # (1,Da)

        raw_action_np = raw_action.detach().cpu().numpy()  # (1,14)

        if self.dp_action_denorm:
            action_np = self.post_process(raw_action_np.copy())
        else:
            action_np = raw_action_np.copy()

        # ===== debug dump：保存一份在线 I/O 样本 =====
        self._debug_dump({
            "t": int(self.t),
            "img_key": img_key,
            "img_dtype": str(img_np.dtype),
            "img_shape": tuple(img_np.shape),
            "img_min": float(np.min(img_np_f)),
            "img_max": float(np.max(img_np_f)),
            "already_imagenet": bool(already_imagenet),

            "qpos_raw_min": float(np.min(raw_qpos)),
            "qpos_raw_max": float(np.max(raw_qpos)),
            "qpos_used_min": float(np.min(qpos_used)),
            "qpos_used_max": float(np.max(qpos_used)),


            "raw_action_min": float(np.min(raw_action_np)),
            "raw_action_max": float(np.max(raw_action_np)),
            "post_action_min": float(np.min(action_np)),
            "post_action_max": float(np.max(action_np)),
            "raw_action": raw_action_np[0],
            "post_action": action_np[0],
        })
        # ===== debug dump 结束 =====

        self.t += 1
        return action_np

