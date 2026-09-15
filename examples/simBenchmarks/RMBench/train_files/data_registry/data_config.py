"""RMBench benchmark — dataset mixtures.

RMBench data converts to the exact same 14-dim dual-arm Agilex ``abs_qpos``
layout as Robotwin (see ``train_files/convert_rmbench_to_lerobot.py``), so the
mixtures below reuse Robotwin's existing data configs instead of defining a new
``ROBOT_TYPE_CONFIG_MAP`` entry:

- ``robotwin50`` — action chunk 50 (``action_indices = range(50)``), min_max on
  joints + binary on grippers. Default for the QwenPI_v3 config in this
  directory (``action_horizon: 50``).
- ``robotwin``   — action chunk 16 variant; swap the third tuple element to
  ``"robotwin"`` (and set ``action_horizon: 16`` in the yaml) to use it.

Auto-discovered by ``starVLA.dataloader.gr00t_lerobot.registry`` at import time
(any ``examples/**/train_files/data_registry/data_config.py`` is merged).
"""

# RMBench (RoboTwin-Platform) demo_clean tasks, 50 episodes each.
RMBENCH_TASKS = [
    "battery_try",
    "blocks_ranking_try",
    "classify_blocks",
    "cover_blocks",
    "observe_and_pickup",
    "place_block_mat",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "storage_blocks",
    "swap_blocks",
    "swap_T",
]

DATASET_NAMED_MIXTURES = {
    # All 12 tasks, one LeRobot dataset dir per task under data_root_dir.
    "rmbench_all": [
        ("battery_try", 1.0, "robotwin50"),
        ("blocks_ranking_try", 1.0, "robotwin50"),
        ("classify_blocks", 1.0, "robotwin50"),
        ("cover_blocks", 1.0, "robotwin50"),
        ("observe_and_pickup", 1.0, "robotwin50"),
        ("place_block_mat", 1.0, "robotwin50"),
        ("press_button", 1.0, "robotwin50"),
        ("put_back_block", 1.0, "robotwin50"),
        ("rearrange_blocks", 1.0, "robotwin50"),
        ("storage_blocks", 1.0, "robotwin50"),
        ("swap_blocks", 1.0, "robotwin50"),
        ("swap_T", 1.0, "robotwin50"),
    ],
    # Single-task mixture for smoke runs.
    "rmbench_rearrange_blocks": [
        ("rearrange_blocks", 1.0, "robotwin50"),
    ],
}

# One single-task mixture per task, named ``rmbench_<task>`` — per RMBench
# protocol each task is trained as its own policy, not as part of a mixture.
for _task in RMBENCH_TASKS:
    DATASET_NAMED_MIXTURES.setdefault(f"rmbench_{_task}", [(_task, 1.0, "robotwin50")])
