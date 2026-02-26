import argparse, os
import numpy as np
import h5py

# 确保能 import 到你的 DP
# 方式1：把 dp_policy.py 放到 repo 根目录或 tools 同级
# 方式2：运行时设置 PYTHONPATH=.
from dp_policy import DP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dp_cfg", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--use_stats", type=str, default="true")  # true/false
    args = ap.parse_args()

    dp = DP({
        "device": args.device,
        "ckpt_dir": args.ckpt_dir,
        "diffusion_policy_cfg": args.dp_cfg,
        "temporal_agg": False,
        "dp_camera_key": "cam_high",
    })

    with h5py.File(args.hdf5, "r") as f:
        qpos = f["observations/qpos"][:]                    # (T,14)
        imgs = f["observations/images/cam_high"][:]         # (T,H,W,3) uint8
        act_gt = f["action"][:]                             # (T,14)

    T = min(len(qpos), args.num_steps)
    preds, gts = [], []

    for t in range(T):
        obs = {"qpos": qpos[t], "cam_high": imgs[t]}
        a = dp.get_action(obs)[0]          # (14,)
        preds.append(a)
        gts.append(act_gt[t])

    preds = np.asarray(preds)
    gts = np.asarray(gts)

    mse = np.mean((preds - gts) ** 2)
    mae = np.mean(np.abs(preds - gts))

    print("pred range:", float(preds.min()), float(preds.max()), "mean", float(preds.mean()), "std", float(preds.std()))
    print("gt   range:", float(gts.min()), float(gts.max()), "mean", float(gts.mean()), "std", float(gts.std()))
    print("MSE:", float(mse), "MAE:", float(mae))

if __name__ == "__main__":
    main()
