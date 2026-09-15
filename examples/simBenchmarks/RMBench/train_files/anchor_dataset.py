"""Episode-aware RMBench anchor loading, isolated from the generic WM dataset."""
from pathlib import Path
from collections import OrderedDict

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP, DATASET_NAMED_MIXTURES
from starVLA.model.modules.vlm.working_memory import history_delta_indices
from examples.simBenchmarks.RMBench.anchor_memory import memory_contract, observation_metadata, resize_memory


class RMBenchAnchorDataset(LeRobotSingleDataset):
    def __init__(self, *args, anchor_contract, **kwargs):
        self.anchor_contract = anchor_contract
        self._episode_anchors = OrderedDict()
        super().__init__(*args, **kwargs)

    def get_step_data(self, trajectory_id, base_index):
        data = super().get_step_data(trajectory_id, base_index)
        # get_video(..., 0) clamps all negative history deltas to the REAL
        # episode start. No dependence on the sampled chunk/window start.
        if trajectory_id not in self._episode_anchors:
            self._episode_anchors[trajectory_id] = self.get_video(trajectory_id, "video.cam_high", 0)[-1].copy()
            if len(self._episode_anchors) > 64:
                self._episode_anchors.popitem(last=False)
        self._episode_anchors.move_to_end(trajectory_id)
        data["rmbench_anchor_image"] = self._episode_anchors[trajectory_id]
        data["rmbench_memory"] = observation_metadata(base_index, self.anchor_contract, trajectory_id)
        return data

    def _pack_sample(self, data):
        sample = super()._pack_sample(data)  # preserves the original FOUR WM images
        sample["image"].insert(0, resize_memory(data["rmbench_anchor_image"], self.anchor_contract["anchor_image_size"]))
        sample["rmbench_memory"] = data["rmbench_memory"]
        return sample


def get_vla_dataset(data_cfg, wm_cfg, anchor_cfg, **kwargs):
    contract = memory_contract({"working_memory": wm_cfg, "rmbench_anchor": anchor_cfg})
    if contract is None or not str(data_cfg.data_mix).startswith("rmbench_"):
        raise ValueError("rmbench_anchor dataset requires enabled anchor and an rmbench_* data mix")
    if data_cfg.get("delete_pause_frame", False):
        raise ValueError("RMBench anchor requires original episode indexing; delete_pause_frame must be false")
    datasets, seen = [], set()
    for name, weight, robot_type in DATASET_NAMED_MIXTURES[data_cfg.data_mix]:
        if (name, robot_type) in seen:
            continue
        seen.add((name, robot_type))
        if robot_type != "robotwin50":
            raise ValueError(f"Unsupported RMBench embodiment: {robot_type}")
        cfg = ROBOT_TYPE_CONFIG_MAP[robot_type]
        modalities = cfg.modality_config()
        modalities["video"].delta_indices = history_delta_indices(contract["history_frames"], contract["history_stride"])
        dataset = RMBenchAnchorDataset(
            dataset_path=Path(data_cfg.data_root_dir) / name,
            modality_configs=modalities, transforms=cfg.transform(),
            embodiment_tag=cfg.embodiment_tag, data_cfg=data_cfg,
            video_backend=data_cfg.get("video_backend", "torchvision_av"),
            anchor_contract=contract,
        )
        dataset.wm_cfg = wm_cfg
        datasets.append((dataset, weight))
    return LeRobotMixtureDataset(datasets, data_cfg=data_cfg, **kwargs)
