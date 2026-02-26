import os

# === RoboTwin: 让 MuJoCo 用 EGL 后端，方便无显示环境训练 ===
os.environ["MUJOCO_GL"] = "egl"

import torch
import numpy as np
import pickle
import argparse

import matplotlib
# === RoboTwin: 不用交互式后端（Agg），避免训练时弹窗 ===
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import json

# 你自己的数据整理 / NashMTL 模块
from utils_data_curation import make_fixed_horizon_collate
from methods.weight_methods import WeightMethods

# 常量：DT / PUPPET 用 RoboTwin 的 constants，
# FUSION_MAP 保留在你自定义的 constants_data_curation 里
from constants import DT, PUPPET_GRIPPER_JOINT_OPEN, SIM_TASK_CONFIGS, FUSION_MAP

from utils_data_curation import load_data, load_data_dual
from utils_data_curation import sample_box_pose, sample_insertion_pose
from utils_data_curation import compute_dict_mean, set_seed, detach_dict

# RoboTwin 中 ACT 的封装叫 act_policy.py
from act_policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        raise NotImplementedError('Only sim tasks are supported in this script.')
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names}
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone,
                         'num_queries': 1, 'camera_names': camera_names}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        # === 新增：与 RoboTwin 一致，用于控制保存频率 ===
        'save_freq': args.get('save_freq', 6000),
    }

    if is_eval:
        ckpt_names = [f'policy_last.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    T = policy_config['num_queries']
    collate = make_fixed_horizon_collate(T)

    # 按 FUSION_MAP 自动切换单/双目录
    if task_name in FUSION_MAP:
        extra_task = FUSION_MAP[task_name]['extra_task']
        assert extra_task in SIM_TASK_CONFIGS, f'FUSION_MAP 指向的任务未在 SIM_TASK_CONFIGS 中：{extra_task}'
        dataset_dir2 = SIM_TASK_CONFIGS[extra_task]['dataset_dir']
        print('Using dual-dataset mode.')

        train_loader, val_loader, stats, is_sim = load_data_dual(
            dataset_dir, dataset_dir2, camera_names,
            batch_size_train, batch_size_val,
            collate_fn=collate
        )
    else:
        train_loader, val_loader, stats, is_sim = load_data(
            dataset_dir, num_episodes, camera_names,
            batch_size_train, batch_size_val,
            collate_fn=collate
        )

    action_dim = int(np.array(stats['action_mean']).shape[-1])
    qpos_dim = int(np.array(stats['qpos_mean']).shape[-1])
    assert action_dim == qpos_dim, "action/qpos 维度不一致"
    config['state_dim'] = action_dim
    policy_config['state_dim'] = action_dim  # 若 ACTPolicy 支持该字段，就一起传下去

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 使用修正后的变量名
    best_ckpt_info = train_bc(train_loader, val_loader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load stats first to infer correct state/action dims
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # infer dims from stats (must match training)
    action_dim = int(np.array(stats['action_mean']).shape[-1])
    qpos_dim = int(np.array(stats['qpos_mean']).shape[-1])
    assert action_dim == qpos_dim, "action/qpos 维度不一致（eval 阶段）"

    # 用 stats 推出来的维度覆盖 config & policy_config
    # 这样 ACTPolicy 构造出来的网络结构就和训练时、ckpt 里的完全一致（都是 14）
    config['state_dim'] = action_dim
    policy_config['state_dim'] = action_dim
    state_dim = action_dim  # 覆盖本地变量，后面 all_time_actions 也会用到这个

    # load policy
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        # from aloha_scripts.robot_utils import move_grippers # requires aloha
        # from aloha_scripts.real_env import make_real_env # requires aloha
        # env = make_real_env(init_node=True)
        # env_max_reward = 0
        raise NotImplementedError('Real-robot eval is not supported in this script.')
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            # move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return

def _forward_single(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)

# 3) 统一前向：支持 (data1, None) 或 (data1, data2)
# 按有效 token 数自然加权
# def forward_pass(data, policy):
#     # 形如 ((img1, q1, a1, m1), (img2, q2, a2, m2)) 的批
#     if isinstance(data, (list, tuple)) and len(data) == 2 and isinstance(data[0], (list, tuple)):
#         d1, d2 = data
#         def to_cuda(d):
#             img, q, a, m = d
#             return img.cuda(), q.cuda(), a.cuda(), m.cuda()
#
#         img1, q1, a1, m1 = to_cuda(d1)
#         if d2 is None:
#             return policy(q1, img1, a1, m1)
#
#         img2, q2, a2, m2 = to_cuda(d2)
#
#         # 关键：在 batch 维拼起来 => 单次前向、单次求均值
#         images = torch.cat([img1, img2], dim=0)
#         qpos   = torch.cat([q1,  q2],  dim=0)
#         action = torch.cat([a1,  a2],  dim=0)
#         mask   = torch.cat([m1,  m2],  dim=0)
#         return policy(qpos, images, action, mask)
#
#     # 兼容只返回单份数据的情况
#     if isinstance(data, (list, tuple)) and len(data) == 4:
#         img, q, a, m = data
#         return policy(q.cuda(), img.cuda(), a.cuda(), m.cuda())
#
#     raise ValueError('Unexpected data format for forward_pass.')

# imitate_episodes_data_curation.py

def forward_pass(data, policy, alpha_for_val=None):
    """
    双目录：分别前向，
        若 alpha_for_val 不为 None，则用该权重加权两边的 loss（Nash 学到的权重）；
        否则按有效 token 数加权。
    单目录：保持原逻辑。
    返回的字典里保留加权后的 'loss'，并额外带上分支日志项（不参与反传）。
    """
    def to_cuda(d):
        img, q, a, m = d
        return img.cuda(), q.cuda(), a.cuda(), m.cuda()

    # 情况B：双目录 batch -> (四元组, 四元组)
    if isinstance(data, (list, tuple)) and len(data) == 2 and isinstance(data[0], (list, tuple)):
        d1, d2 = data
        img1, q1, a1, m1 = to_cuda(d1)
        if d2 is None:
            return policy(q1, img1, a1, m1)

        img2, q2, a2, m2 = to_cuda(d2)

        # 两次前向
        out1 = policy(q1, img1, a1, m1)  # 字典，含 out1['loss'] 等
        out2 = policy(q2, img2, a2, m2)

        # 在这里打印两个分支的损失（使用 detach().item() 获取标量）
        try:
            l1 = out1['loss'].detach().item()
        except Exception:
            l1 = float(out1['loss'].detach())
        try:
            l2 = out2['loss'].detach().item()
        except Exception:
            l2 = float(out2['loss'].detach())
        # print('------------------------------------------')
        print(f'[forward_pass] loss_branch1={l1:.6f}, loss_branch2={l2:.6f}')
        # print('------------------------------------------')

        # 有效token数（mask=True 表示pad，所以取 ~mask）
        eff1 = (~m1).float().sum()  # 标量
        eff2 = (~m2).float().sum()
        wsum = eff1 + eff2 + 1e-8

        # === 关键：决定验证时的加权系数 w1 / w2 ===
        # 若提供了 alpha_for_val（来自训练阶段的 Nash 平均权重），优先使用它；
        # 否则退化为按有效 token 数加权。
        if alpha_for_val is not None:
            if isinstance(alpha_for_val, torch.Tensor):
                alpha_local = alpha_for_val.detach().to(out1['loss'].device)
                w1 = alpha_local[0]
                w2 = alpha_local[1]
            else:
                # 兼容 list / tuple / numpy
                w1 = torch.as_tensor(alpha_for_val[0], device=out1['loss'].device, dtype=out1['loss'].dtype)
                w2 = torch.as_tensor(alpha_for_val[1], device=out1['loss'].device, dtype=out1['loss'].dtype)
        else:
            # 默认：按有效 token 数加权
            w1 = eff1 / wsum
            w2 = eff2 / wsum

        # 标量项做加权平均（loss / l1 / kl 用 w1,w2），其余项做简单平均
        merged = {}
        for k in out1:
            v1, v2 = out1[k], out2[k]
            if torch.is_tensor(v1) and v1.dim() == 0 and torch.is_tensor(v2) and v2.dim() == 0:
                if k in ('loss', 'l1', 'kl'):
                    merged[k] = w1 * v1 + w2 * v2
                else:
                    merged[k] = 0.5 * (v1 + v2)
            else:
                merged[k] = 0.5 * (v1 + v2)

        # 仅用于日志观测（不参与反传）
        merged['loss_branch1'] = out1['loss'].detach()
        merged['loss_branch2'] = out2['loss'].detach()
        merged['eff_tokens_branch1'] = eff1.detach()
        merged['eff_tokens_branch2'] = eff2.detach()
        return merged

    # 情况A：单目录 batch -> 四元组
    if isinstance(data, (list, tuple)) and len(data) == 4:
        img, q, a, m = data
        return policy(q.cuda(), img.cuda(), a.cuda(), m.cuda())

    raise ValueError('Unexpected data format for forward_pass.')


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    # === 新增 ===
    save_freq = config.get('save_freq', 6000)

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    # === [NEW] optional load latest checkpoint & resume epoch index ===
    latest_idx = 0

    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 1:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input().strip().lower()

        if load_ckpt == "y":
            ckpt_epochs = [
                int(f.split("_")[2])
                for f in os.listdir(ckpt_dir)
                if f.startswith("policy_epoch_") and f.endswith(f"_seed_{seed}.ckpt")
                ]

            if len(ckpt_epochs) > 0:
                latest_idx = max(ckpt_epochs)
                ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt")
                print(f"Loading checkpoint from {ckpt_path}")
                loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                print(loading_status)
            else:
                print("No policy_epoch_* checkpoints found; start from scratch.")
        else:
            print("Not loading checkpoint.")
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # === 新增：检测是否是双目录数据，并构造 Nash-MTL ===
    device = next(policy.parameters()).device
    has_dual = hasattr(train_dataloader.dataset, "dataset_dir2") and \
               (getattr(train_dataloader.dataset, "dataset_dir2") is not None)

    if has_dual:
        print("[NashMTL] Detected dual-dataset training, use Nash-MTL for loss weighting.")
        n_tasks = 2  # 两个目录 => 两个任务
        weight_method = WeightMethods(
            method="nashmtl",
            n_tasks=n_tasks,
            device=device,
            max_norm=1.0,
            update_weights_every=1,  # 每个 step 都重新算一次权重
        )
    else:
        print("[NashMTL] Single-dataset training, fall back to original scalar loss.")
        weight_method = None
    # ===============================================

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    best_meta_path = os.path.join(ckpt_dir, f"best_info_seed_{seed}.json")
    if os.path.isfile(best_meta_path):
        try:
            with open(best_meta_path, "r") as f:
                best_meta = json.load(f)
            prev_best_epoch = int(best_meta.get("best_epoch"))
            prev_best_val = float(best_meta.get("min_val_loss"))
            prev_best_path = os.path.join(ckpt_dir, f"policy_epoch_{prev_best_epoch}_seed_{seed}.ckpt")
            if os.path.isfile(prev_best_path):
                best_state = torch.load(prev_best_path, map_location="cpu")
                min_val_loss = prev_best_val
                best_ckpt_info = (prev_best_epoch, prev_best_val, best_state)
                print(f"[resume] Found historical best: epoch {prev_best_epoch}, val {prev_best_val:.6f}")
        except Exception as e1:
            print(f"[resume] best_info read failed: {e1}")

    for epoch in tqdm(range(latest_idx, num_epochs)):
        print(f'\nEpoch {epoch}')

        # ========= 1. 训练阶段：收集本 epoch 所有 batch 的 alpha =========
        policy.train()
        optimizer.zero_grad()
        epoch_alphas = []  # 用于统计本轮的 Nash 权重均值

        for batch_idx, data in enumerate(train_dataloader):

            # 情况1：没有 Nash（单目录），保持原来的 forward_pass 逻辑
            if weight_method is None:
                forward_dict = forward_pass(data, policy)
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
                continue

            # 情况2：双目录 + Nash-MTL，多任务更新
            # data 结构是 ((img1, q1, a1, m1), (img2, q2, a2, m2))
            (d1, d2) = data
            out1 = _forward_single(d1, policy)  # dict，含 'loss', 'l1', 'kl' ...
            out2 = _forward_single(d2, policy)

            # loss1 = out1['loss']
            # loss2 = out2['loss']
            # losses = torch.stack([loss1, loss2])  # [2]
            #
            # # 共享参数：这里直接用整个 policy 的参数作为“shared_parameters”
            # shared_parameters = [p for p in policy.parameters() if p.requires_grad]
            #
            # # 用 Nash-MTL 做一次 backward（内部已经调用 weighted_loss.backward()）
            # total_loss, extra = weight_method.backward(
            #     losses=losses,
            #     shared_parameters=shared_parameters,
            # )

            loss1 = out1['loss']
            loss2 = out2['loss']

            # 人为偏好：给任务 2 打个折扣（比如 0.7）
            bias1 = 1
            bias2 = 0.7

            scaled_losses = torch.stack([bias1 * loss1, bias2 * loss2])

            shared_parameters = [p for p in policy.parameters() if p.requires_grad]

            total_loss, extra = weight_method.backward(
                losses=scaled_losses,  # 👈 这里用缩放后的 loss 参与 Nash 博弈
                shared_parameters=shared_parameters,
            )

            optimizer.step()
            optimizer.zero_grad()

            # 记录本 batch 的 Nash 权重
            alpha = extra['weights']  # tensor, shape [2]
            epoch_alphas.append(alpha.detach())

            # 为了后面画曲线/打印，把当前 batch 的统计信息整理成一个 dict
            log_dict = {}
            for k in out1:
                v1, v2 = out1[k], out2[k]
                if torch.is_tensor(v1) and v1.dim() == 0 and torch.is_tensor(v2) and v2.dim() == 0:
                    # 标量指标：loss / l1 / kl 用 Nash 权重；其他用简单平均
                    if k in ('loss', 'l1', 'kl'):
                        log_dict[k] = (alpha[0] * v1 + alpha[1] * v2)
                    else:
                        log_dict[k] = 0.5 * (v1 + v2)
                else:
                    # 非标量（几乎没有），简单平均或按需处理
                    log_dict[k] = 0.5 * (v1 + v2)

            # 额外记录每个分支的 loss 和 Nash 权重（只用于日志）
            log_dict['loss_branch1'] = loss1
            log_dict['loss_branch2'] = loss2
            log_dict['nash_weight1'] = alpha[0]
            log_dict['nash_weight2'] = alpha[1]

            train_history.append(detach_dict(log_dict))

        # 本 epoch 训练指标（照旧，用 train_history 做切片）
        e = epoch - latest_idx
        epoch_train_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e: (batch_idx + 1) * (e + 1)]
        )
        epoch_train_loss = epoch_train_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_train_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # ========= 2. 统计本轮 α 的均值，用于验证阶段 =========
        alpha_for_val = None
        if weight_method is not None and len(epoch_alphas) > 0:
            # shape [2]，不需要梯度
            alpha_for_val = torch.stack(epoch_alphas, dim=0).mean(dim=0).detach()

        # ========= 3. 验证阶段：用 alpha_for_val 聚合 val loss =========
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx_val, data in enumerate(val_dataloader):
                if alpha_for_val is not None:
                    forward_dict = forward_pass(data, policy, alpha_for_val=alpha_for_val)
                else:
                    forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_val_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_val_summary)

            epoch_val_loss = epoch_val_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                # 同步落盘 best 元信息
                try:
                    with open(best_meta_path, "w") as f:
                        json.dump(
                            {"best_epoch": int(epoch),
                             "min_val_loss": float(min_val_loss),
                             "seed": int(seed)}, f)
                except Exception as e1:
                    print(f"[best_info save failed] {e1}")

        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_val_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # ========= 4. 周期性存 ckpt + 画图（保持不变） =========
        if epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    try:
        with open(best_meta_path, "w") as f:
            json.dump(
                {"best_epoch": int(best_epoch),
                 "min_val_loss": float(min_val_loss),
                 "seed": int(seed)}, f)
    except Exception as e1:
        print(f"[best_info save failed at end] {e1}")

    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # 只画训练 & 验证都存在的键
    common_keys = set(train_history[0].keys()) & set(validation_history[0].keys())

    for key in common_keys:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument("--state_dim", action="store", type=int,
                        help="state dim (ignored, kept for compatibility)", required=False)
    parser.add_argument("--save_freq", action="store", type=int,
                        help="save ckpt frequency", required=False, default=6000)
    
    main(vars(parser.parse_args()))
