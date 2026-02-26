import h5py

def inspect_hdf5_structure(file_path, max_depth=2):
    """
    打印HDF5文件的层级结构、每个数据集的shape和dtype。
    """
    def print_attrs(name, obj, depth=0):
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}- {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}+ {name}/")
            if depth < max_depth:
                for key in obj.keys():
                    print_attrs(f"{name}/{key}", obj[key], depth + 1)
    with h5py.File(file_path, "r") as f:
        print(f"\n📂 Inspecting file: {file_path}")
        for key in f.keys():
            print_attrs(key, f[key])

inspect_hdf5_structure("/home/mn/DemoSpeedup/aloha/data/sim_beat_block_hammer_act_entropy/episode_0.hdf5")
# inspect_hdf5_structure("/home/mn/RoboTwin-main/policy/ACT/processed_data/sim_beat_block_hammer/demo_clean/episode_0.hdf5")
# inspect_hdf5_structure("/home/mn/RoboTwin-main/policy/ACT/processed_data/sim_beat_block_hammer/demo_clean_2x/episode_0.hdf5")
# inspect_hdf5_structure("/home/mn/RoboTwin-main/policy/ACT/processed_data/sim_beat_block_hammer/demo_clean_2x/episode_1.hdf5")
# inspect_hdf5_structure("/home/mn/DemoSpeedup/aloha/data/sim_transfer_cube_human_origin/episode_20.hdf5")
# inspect_hdf5_structure("/home/mn/act-main/datas/sim_transfer_cube_human_2x_200/episode_0.hdf5")