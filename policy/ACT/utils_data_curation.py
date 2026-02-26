import numpy as np
import torch
import os
import h5py
import glob
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

# utils_data_curation.py
def make_fixed_horizon_collate(T: int):
    def _single_collate(batch):
        # batch: List[(images, qpos, action, is_pad)]
        images, qpos, actions, masks = zip(*batch)
        images = torch.stack([torch.as_tensor(x) for x in images], 0)  # [B,K,C,H,W]
        qpos   = torch.stack([torch.as_tensor(x) for x in qpos],   0)  # [B,14]

        a_out, m_out = [], []
        for a, m in zip(actions, masks):
            a = torch.as_tensor(a)  # [L,14]
            m = torch.as_tensor(m)  # [L]
            a_t, m_t = a[:T], m[:T]
            if a_t.shape[0] < T:
                pad = T - a_t.shape[0]
                a_t = torch.cat([a_t, torch.zeros(pad, a.shape[1], dtype=a.dtype)], 0)
                m_t = torch.cat([m_t.to(torch.bool), torch.ones(pad, dtype=torch.bool)], 0)
            a_out.append(a_t); m_out.append(m_t)
        actions = torch.stack(a_out, 0)  # [B,T,14]
        masks   = torch.stack(m_out, 0)  # [B,T]
        return images, qpos, actions, masks

    def collate(batch):
        elem = batch[0]
        # 情况A：单目录 -> 四元组
        if isinstance(elem, (list, tuple)) and len(elem) == 4:
            return _single_collate(batch)
        # 情况B：双目录 -> 二元组(四元组, 四元组)
        if isinstance(elem, (list, tuple)) and len(elem) == 2 and isinstance(elem[0], (list, tuple)):
            d1_list, d2_list = zip(*batch)
            b1 = _single_collate(list(d1_list))
            # 若可能存在 data2=None，也可在此加判空
            b2 = _single_collate(list(d2_list))
            return b1, b2
        raise ValueError("Unexpected batch structure for collate.")
    return collate

def episode_filename(episode_id: int) -> str:
    # 适配你的文件名：episode_0.hdf5, episode_1.hdf5, ...
    return f"episode_{episode_id}.hdf5"


class EpisodicDataset(torch.utils.data.Dataset):
    """
    统一数据集：
    - 若 dataset_dir2 为 None：返回 (data1, None)
    - 若 dataset_dir2 非 None：返回 (data1, data2)
    data* 结构为 (image_data, qpos_data, action_data, is_pad)
    """
    def __init__(self, episode_ids, dataset_dir1, dataset_dir2, camera_names, norm_stats):
        super().__init__()
        self.episode_ids = list(episode_ids)
        self.dataset_dir1 = dataset_dir1
        self.dataset_dir2 = dataset_dir2  # 可为 None
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None  # 第一次读取时确定

    def __len__(self):
        return len(self.episode_ids)

    def _load_one(self, dataset_dir, episode_id):
        dataset_path = os.path.join(dataset_dir, episode_filename(episode_id))
        with h5py.File(dataset_path, 'r') as root:
            is_sim = bool(root.attrs.get('sim', True))
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]

            # 独立随机起点
            start_ts = np.random.choice(episode_len)

            # 起点观测
            qpos = root['/observations/qpos'][start_ts]
            # _ = root['/observations/qvel'][start_ts]  # 若下游不用，可忽略
            image_dict = {cam: root[f'/observations/images/{cam}'][start_ts] for cam in self.camera_names}

            # 动作序列（仿真：从 start_ts；实机：从 start_ts-1）
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

        if self.is_sim is None:
            self.is_sim = is_sim

        # pad 动作到 episode_len
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len, dtype=np.float32)
        is_pad[action_len:] = 1.0

        # 拼相机维 (K,H,W,C) -> (K,C,H,W)
        all_cam_images = np.stack([image_dict[c] for c in self.camera_names], axis=0)
        image_data = torch.from_numpy(all_cam_images).permute(0, 3, 1, 2).float() / 255.0
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # 归一化
        action_data = (action_data - torch.as_tensor(self.norm_stats["action_mean"]).float()) / torch.as_tensor(self.norm_stats["action_std"]).float()
        qpos_data = (qpos_data - torch.as_tensor(self.norm_stats["qpos_mean"]).float()) / torch.as_tensor(self.norm_stats["qpos_std"]).float()

        return image_data, qpos_data, action_data, is_pad

    def __getitem__(self, index):
        episode_id = int(self.episode_ids[index])
        data1 = self._load_one(self.dataset_dir1, episode_id)
        if self.dataset_dir2 is None:
            # 单目录：直接返回 data1（四元组）
            return data1
        else:
            data2 = self._load_one(self.dataset_dir2, episode_id)
            return data1, data2


def _list_episode_ids(dataset_dir):
    files = glob.glob(os.path.join(dataset_dir, 'episode_*.hdf5'))
    ids = []
    for p in files:
        base = os.path.basename(p)
        try:
            ids.append(int(base.split('_')[1].split('.')[0]))
        except Exception:
            pass
    return sorted(ids)

def get_norm_stats(dataset_dir1, ids1, dataset_dir2=None, ids2=None):
    all_qpos = []
    all_action = []

    # 目录1
    for episode_id in ids1:
        p = os.path.join(dataset_dir1, episode_filename(episode_id))  # 原来是 f'episode_{episode_id}.hdf5'
        with h5py.File(p, 'r') as f:
            qpos = f['/observations/qpos'][()]
            action = f['/action'][()]
        all_qpos.append(torch.from_numpy(qpos))
        all_action.append(torch.from_numpy(action))

        # 目录2（可选）
    if dataset_dir2 is not None and ids2 is not None:
        for episode_id in ids2:
            p = os.path.join(dataset_dir2, episode_filename(episode_id))  # 原来是 f'episode_{episode_id}.hdf5'
            with h5py.File(p, 'r') as f:
                qpos = f['/observations/qpos'][()]
                action = f['/action'][()]
            all_qpos.append(torch.from_numpy(qpos))
            all_action.append(torch.from_numpy(action))

    all_qpos = torch.cat(all_qpos, dim=0)       # (sum_T, Dq)
    all_action = torch.cat(all_action, dim=0)   # (sum_T, Da)

    action_mean = all_action.mean(dim=0, keepdim=False)
    action_std = all_action.std(dim=0, keepdim=False).clamp_(1e-2)
    qpos_mean = all_qpos.mean(dim=0, keepdim=False)
    qpos_std = all_qpos.std(dim=0, keepdim=False).clamp_(1e-2)

    return {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": all_qpos[0].numpy(),
    }

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, collate_fn=None):
    print(f'\nData from: {dataset_dir}\n')
    all_ids = _list_episode_ids(dataset_dir)
    if len(all_ids) == 0:
        raise RuntimeError('Dataset dir is empty.')

    shuffled = np.array(all_ids)[np.random.permutation(len(all_ids))]
    train_ratio = 0.8
    split = int(train_ratio * len(shuffled))
    train_ids = shuffled[:split]
    val_ids = shuffled[split:]

    norm_stats = get_norm_stats(dataset_dir, all_ids)

    train_dataset = EpisodicDataset(train_ids, dataset_dir, None, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_ids, dataset_dir, None, camera_names, norm_stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                              pin_memory=True, num_workers=1, prefetch_factor=1,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True,
                            pin_memory=True, num_workers=1, prefetch_factor=1,
                            collate_fn=collate_fn)

    _ = train_dataset[0]  # 触发 is_sim
    return train_loader, val_loader, norm_stats, train_dataset.is_sim

def load_data_dual(dataset_dir1, dataset_dir2, camera_names, batch_size_train, batch_size_val, collate_fn=None):
    print(f'\nData from two dirs:\n  {dataset_dir1}\n  {dataset_dir2}\n')
    ids1 = set(_list_episode_ids(dataset_dir1))
    ids2 = set(_list_episode_ids(dataset_dir2))
    common_ids = sorted(ids1 & ids2)
    if len(common_ids) == 0:
        raise RuntimeError('No common episode ids between the two dataset dirs.')

    shuffled = np.array(common_ids)[np.random.permutation(len(common_ids))]
    train_ratio = 0.8
    split = int(train_ratio * len(shuffled))
    train_ids = shuffled[:split]
    val_ids = shuffled[split:]

    norm_stats = get_norm_stats(dataset_dir1, common_ids, dataset_dir2, common_ids)

    train_ds = EpisodicDataset(train_ids, dataset_dir1, dataset_dir2, camera_names, norm_stats)
    val_ds = EpisodicDataset(val_ids, dataset_dir1, dataset_dir2, camera_names, norm_stats)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1, collate_fn=collate_fn)

    _ = train_ds[0]
    return train_loader, val_loader, norm_stats, train_ds.is_sim

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
