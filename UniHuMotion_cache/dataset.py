import os
import pickle
from pathlib import Path

import numpy as np
import torch
from utils import Reader_UniHuMotion


REPO_ROOT = Path(__file__).resolve().parents[1]


def dataset_short_name(name):
    if not name.startswith("UHM_"):
        raise ValueError(f"Expected a UniHuMotion dataset name like 'UHM_NBA', got {name!r}.")
    return name[4:]


class MultiPersonTrajPoseDataset(torch.utils.data.Dataset):
    def __init__(self, name, split="train", track_size=21, track_cutoff=9, segmented=True,
                 add_flips=False, frequency=1, chunk_size=4, obs_len=None, max_agents=8):
        self.name = name
        self.split = split
        self.track_size = track_size
        self.track_cutoff = obs_len if obs_len is not None else track_cutoff
        self.frequency = frequency
        self.data_chunk_size = chunk_size
        self.max_agents = max_agents
        self.datalist = []
        self.dataset_idx = {}
        self.load_data()

    def load_data(self):
        self.datalist = []
        self.dataset_idx = {}
        joint_and_mask_temp = []

        data_name = dataset_short_name(self.name)
        cache_dir = REPO_ROOT / "data" / "cache" / data_name / self.split
        cache_dir.mkdir(parents=True, exist_ok=True)

        uhm_dir = REPO_ROOT / "data" / "UniHuMotion" / self.name
        train_scenes, _, _ = prepare_data(uhm_dir, subset=self.split, sample=1.0, goals=False, dataset_name=self.name)

        loaded_sample = 0
        chunk_idx = 0
        for _, _, paths in train_scenes:
            scene_train = Reader_UniHuMotion.paths_to_xy(paths)
            scene_train = drop_ped_with_missing_frame(scene_train, obs=self.track_cutoff)
            scene_train = drop_distant(scene_train, obs=self.track_cutoff, max_num_peds=self.max_agents)

            t, n = scene_train.shape[0], scene_train.shape[1]
            traj_4_col = np.pad(scene_train[:, :, :2].reshape(t, n, -1, 2), ((0, 0), (0, 0), (0, 0), (0, 2)), mode="constant")
            bb3d_4_col = scene_train[:, :, 2:6].reshape(t, n, -1, 4)
            bb2d_4_col = scene_train[:, :, 6:10].reshape(t, n, -1, 4)
            pose_3d_4_col = np.pad(
                np.transpose(scene_train[:, :, 10:127].reshape(t, n, -1, 39), (0, 1, 3, 2)),
                ((0, 0), (0, 0), (0, 0), (0, 1)),
                mode="constant",
            )
            pose_2d_4_col = np.pad(
                np.transpose(scene_train[:, :, 127:205].reshape(t, n, -1, 39), (0, 1, 3, 2)),
                ((0, 0), (0, 0), (0, 0), (0, 2)),
                mode="constant",
            )

            scene_train_real = np.concatenate((traj_4_col, bb3d_4_col, bb2d_4_col, pose_3d_4_col, pose_2d_4_col), axis=2)
            joints = np.asarray(np.transpose(scene_train_real, (1, 0, 2, 3)))
            mask = np.ones(joints.shape[:-1])

            joint_and_mask_temp.append((joints, mask))
            if len(joint_and_mask_temp) == self.data_chunk_size:
                save_data_to_pickle(joint_and_mask_temp, cache_dir / f"{data_name}_idx_{chunk_idx}.pkl")
                chunk_idx += 1
                joint_and_mask_temp = []

            self.datalist.append("0")
            loaded_sample += 1
            if loaded_sample % 1000 == 0:
                print("finished loading", int(loaded_sample / 1000), "k samples")

        if joint_and_mask_temp:
            save_data_to_pickle(joint_and_mask_temp, cache_dir / f"{data_name}_idx_{chunk_idx}.pkl")

        print("loaded", loaded_sample, "samples in total.")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]


class UniHuMotionDataset(MultiPersonTrajPoseDataset):
    def __init__(self, dataset_name, split, chunk_size=4, obs_len=None, max_agents=8):
        super().__init__(
            dataset_name,
            split=split,
            frequency=1,
            chunk_size=chunk_size,
            obs_len=obs_len,
            max_agents=max_agents,
        )


def create_dataset(dataset_name, split="train", chunk_size=4, obs_len=None, max_agents=8):
    if dataset_name.startswith("UHM_"):
        return UniHuMotionDataset(dataset_name, split=split, chunk_size=chunk_size, obs_len=obs_len, max_agents=max_agents)
    raise ValueError(f"Dataset with name '{dataset_name}' not found.")


def get_datasets(datasets_list, chunk_size=4, obs_len=None, max_agents=8):
    return [create_dataset(name, split="train", chunk_size=chunk_size, obs_len=obs_len, max_agents=max_agents) for name in datasets_list]


def save_data_to_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_data_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def drop_ped_with_missing_frame(xy, obs):
    xy_n_t = np.transpose(xy, (1, 0, 2))
    mask = np.ones(xy_n_t.shape[0], dtype=bool)
    for n in range(1, xy_n_t.shape[0]):
        for t in range(obs):
            if np.isnan(xy_n_t[n, t, 0]):
                mask[n] = False
                break
    return np.transpose(xy_n_t[mask], (1, 0, 2))


def drop_distant(xy, obs, max_num_peds=8):
    distance_2 = np.sum(np.square(xy[:obs, :, 0:2] - xy[:obs, 0:1, 0:2]), axis=2)
    smallest_dist_to_ego = np.nanmin(distance_2, axis=0)
    return xy[:, np.argsort(smallest_dist_to_ego)[:max_num_peds]]


def prepare_data(path, subset="train", sample=1.0, goals=True, dataset_name=""):
    all_scenes = []
    subset_path = Path(path) / subset
    if not subset_path.exists():
        raise FileNotFoundError(f"UniHuMotion split directory not found: {subset_path}")

    files = sorted(p for p in subset_path.iterdir() if p.suffix == ".ndjson")
    for file_path in files:
        reader = Reader_UniHuMotion(str(file_path), scene_type="paths")
        scene = [(file_path.stem, scene_id, scene_paths) for scene_id, scene_paths in reader.scenes(sample=sample)]
        all_scenes += scene
    return all_scenes, None, True
