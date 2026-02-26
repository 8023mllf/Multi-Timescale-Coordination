# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython

e = IPython.embed


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)  # will be overridden
    parser.add_argument("--lr_backbone", default=1e-5, type=float)  # will be overridden
    parser.add_argument("--batch_size", default=2, type=int)  # not used
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)  # not used
    parser.add_argument("--lr_drop", default=200, type=int)  # not used
    parser.add_argument(
        "--clip_max_norm",
        default=0.1,
        type=float,  # not used
        help="gradient clipping max norm",
    )

    # Model parameters
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet18",
        type=str,  # will be overridden
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--camera_names",
        default=[],
        type=list,  # will be overridden
        help="A list of camera names",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers",
        default=4,
        type=int,  # will be overridden
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,  # will be overridden
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,  # will be overridden
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,  # will be overridden
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,  # will be overridden
        help="Number of attention heads inside the transformer's attentions",
    )
    # parser.add_argument('--num_queries', required=True, type=int, # will be overridden
    #                     help="Number of query slots")#AGGSIZE
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--state_dim", action="store", type=int, help="state dim", required=True)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    # parser.add_argument('--num_queries',type=int, required=True)
    # parser.add_argument('--actionsByQuery',type=int, required=True)

    return parser

import argparse

def build_ACT_model_and_optimizer(args_override, RoboTwin_Config=None):
    """
    两种使用方式：
    1）训练脚本：RoboTwin_Config is None 且 args_override is None
       -> 从命令行解析参数（保持原行为）

    2）代码调用（比如 ACTPolicy 在 RoboTwin 训练里、BiGym eval 里调用）：
       RoboTwin_Config is None 且 args_override 是 dict
       -> 不读真实命令行，而是喂一份假 argv，再用 args_override 覆盖
    """

    if RoboTwin_Config is None:
        parser = argparse.ArgumentParser(
            "DETR training and evaluation script",
            parents=[get_args_parser()],
        )

        if args_override is None:
            # 训练脚本：老行为，直接读命令行
            args = parser.parse_args()
        else:
            # 现在：库调用模式（无论是 RoboTwin 还是 Bigym）
            dummy_argv = [
                "--ckpt_dir", "dummy",
                "--policy_class", "ACT",
                "--task_name", "dummy_task",
                "--seed", "0",
                "--num_epochs", "1",
                "--state_dim", "1",
            ]
            args = parser.parse_args(dummy_argv)

            # 用 override 里的值覆盖
            for k, v in args_override.items():
                setattr(args, k, v)

            # ★★★ 关键同步逻辑：保证 num_queries 和 chunk_size 都有值 ★★★
            has_chunk = getattr(args, "chunk_size", None) is not None
            has_nq    = getattr(args, "num_queries", None) is not None

            if has_nq and not has_chunk:
                # 只有 num_queries，没有 chunk_size → 用 num_queries 补 chunk_size
                args.chunk_size = args.num_queries
            elif has_chunk and not has_nq:
                # 只有 chunk_size，没有 num_queries → 用 chunk_size 补 num_queries
                args.num_queries = args.chunk_size
            # 如果两者都有，就不动；如果两者都没有，那是配置本身的问题了
    else:
        # 有 RoboTwin_Config 的话，完全按你原来的逻辑走
        args = RoboTwin_Config

    print("build_ACT_model_and_optimizer", args)
    print(args)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    """
    两种使用方式：
    1）训练脚本：args_override is None -> 从命令行解析参数（保持原行为）
    2）库调用/评测：args_override 是 dict/Namespace -> 不读真实命令行，喂 dummy_argv，再用 override 覆盖
    """
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script",
        parents=[get_args_parser()],
    )

    if args_override is None:
        # 老行为：直接读真实命令行（如果你真的用这个函数作为训练入口）
        args = parser.parse_args()
    else:
        # 关键：get_args_parser 里有 required=True 的参数，parse_args([]) 会直接报错
        dummy_argv = [
            "--ckpt_dir", "dummy",
            "--policy_class", "DP",
            "--task_name", "dummy_task",
            "--seed", "0",
            "--num_epochs", "1",
            "--state_dim", "14",
        ]
        args = parser.parse_args(dummy_argv)

        # 兼容 dict / Namespace
        if hasattr(args_override, "items"):
            items = args_override.items()
        else:
            items = vars(args_override).items()

        for k, v in items:
            setattr(args, k, v)

        # 可选：对齐 num_queries/chunk_size（有些地方会用到）
        has_chunk = getattr(args, "chunk_size", None) is not None
        has_nq = getattr(args, "num_queries", None) is not None
        if has_nq and not has_chunk:
            args.chunk_size = args.num_queries
        elif has_chunk and not has_nq:
            args.num_queries = args.chunk_size

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer

