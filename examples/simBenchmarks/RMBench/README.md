# RMBench × starVLA

Training-pipeline integration for [RMBench](https://github.com/RoboTwin-Platform/RMBench)
(Memory-Dependent Robotic Manipulation Benchmark, built on RoboTwin 2.0): 12
dual-arm Agilex tasks, 50 `demo_clean` episodes each (600 episodes total).

> **Scope: training only.** The eval harness (policy server + RMBench sim
> client) is intentionally not included yet — see [Follow-ups](#follow-ups).

## Upstream repo (not vendored)

The RMBench sim/benchmark code is **deliberately not vendored** into starVLA.
Clone it to a sibling location and point `RMBENCH_HOME` at it:

```bash
git clone https://github.com/RoboTwin-Platform/RMBench.git /code/RMBench
export RMBENCH_HOME=/code/RMBench
```

`RMBENCH_HOME` is only needed for data download (and future eval work); the
training pipeline below reads converted data from `/data` directly.

## Data download

Raw data (37 GB, HF `TianxingChen/RMBench`, `data/*/demo_clean/**`) is
downloaded via the upstream repo's `data/_download.py`. On this machine the
checkout redirects output to `/data/starVLA/RMBench`, giving:

```
/data/starVLA/RMBench/data/<task>/demo_clean/
    data/episode{N}.hdf5          # joint_action/vector (T,14) + 3 camera jpeg streams
    instructions/episode{N}.json  # {"seen": [...], "unseen": [...]}
    ...
```

Tasks: `battery_try`, `blocks_ranking_try`, `classify_blocks`, `cover_blocks`,
`observe_and_pickup`, `place_block_mat`, `press_button`, `put_back_block`,
`rearrange_blocks`, `storage_blocks`, `swap_blocks`, `swap_T`.

## Conversion (HDF5 → LeRobot v2.1)

```bash
python examples/simBenchmarks/RMBench/train_files/convert_rmbench_to_lerobot.py \
    --raw-root /data/starVLA/RMBench \
    --out-root /data/starVLA/rmbench_lerobot
    # --tasks rearrange_blocks --max-episodes 2   # optional smoke-test subset
```

One LeRobot dataset per task under `<out_root>/<task>/` (`meta/`, `data/`,
`videos/`, h264 30 fps). Conventions (same as the official Mem-0 converter):

- `state[t] = qpos[t]`, `action[t] = qpos[t+1]` (`abs_qpos`, last frame repeats)
- 14-dim: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)
- Cameras: `head_camera→cam_high`, `left_camera→cam_left_wrist`, `right_camera→cam_right_wrist`
- `meta/modality.json` is written per dataset; `meta/stats_gr00t.json` is
  computed automatically on first training use (rank 0 iterates all parquets
  once — with 12 datasets this takes a few minutes, then it's cached)
- Broken episodes are skipped with a `[convert][skip]` log line — check the
  skip count after the full run

## Registration

`train_files/data_registry/data_config.py` is auto-discovered by
`starVLA.dataloader.gr00t_lerobot.registry` (glob over
`examples/**/train_files/data_registry/`). The 14-dim layout is identical to
Robotwin's Agilex config, so mixtures reuse its data configs directly:

| mixture | content | robot_type (action chunk) |
|---|---|---|
| `rmbench_all` | all 12 tasks × weight 1.0 | `robotwin50` (h=50) |
| `rmbench_rearrange_blocks` | single task, smoke runs | `robotwin50` (h=50) |

For an action-chunk-16 variant, change the third tuple element to `"robotwin"`
and set `framework.action_model.action_horizon: 16` in the yaml.

## Training

```bash
# full run (uses all visible GPUs)
bash examples/simBenchmarks/RMBench/train_files/run_rmbench_train.sh

# smoke run: single task, 20 steps
bash examples/simBenchmarks/RMBench/train_files/run_rmbench_train.sh \
    --datasets.vla_data.data_mix rmbench_rearrange_blocks \
    --trainer.max_train_steps 20 --trainer.save_interval 20

# inspect the resolved command without launching
DRY_RUN=1 bash examples/simBenchmarks/RMBench/train_files/run_rmbench_train.sh
```

Config: `train_files/starvla_qwenpiv3_rmbench.yaml` — QwenPI_v3
(Qwen3-VL-4B + LayerwiseFM action model, `action_horizon: 50`), based on the
RoboDojo PI-v3 recipe. Useful env vars: `rmbench_data_root` (default
`/data/starVLA/rmbench_lerobot`), `NUM_PROCESSES`, `DRY_RUN=1`; extra args are
passed through to `train_starvla.py`.

## Cluster submission (KOALA)

`submit_train_rmbench.sh` (repo root) submits a Kubeflow PyTorchJob that
converts the data in-pod (first run only), pushes the converted dataset back to
the S3 asset bucket, and trains. Converted data and stats caches are reused by
every later job.

**Secrets** — no plaintext in the script. Create `.env.submit` (gitignored) at
the repo root:

```bash
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
WANDB_API_KEY=...
WANDB_ENTITY=...
KOALA_TOKEN=...
```

**First-time setup** (builds the `:rmbench` image — h5py is in
`requirements.txt` — and uploads the 37 GB raw data):

```bash
BUILD_IMAGE=1 bash submit_train_rmbench.sh
```

**Regular submits** (raw data already on S3 — skip the re-check):

```bash
SYNC_RAW=0 bash submit_train_rmbench.sh
```

Pipeline details:

- `SYNC_RAW=1` uploads `/data/starVLA/RMBench/data` →
  `s3://<asset>/<user>/starVLA/datasets/rmbench_raw/data` (incremental).
- In-pod, `train_files/run_rmbench_train_in_pod.sh` checks the converted asset
  (`datasets/rmbench_lerobot`); if tasks are missing it converts from the raw
  asset to `/local-ssd`, pushes the result back to S3, and trains from the
  local copy. Later jobs find all 12 tasks and skip conversion entirely.
- After training, `stats_gr00t.json` / `steps_data_index.pkl` are pushed back
  so later jobs skip the rank-0 stats computation.
- Checkpoints mirror to `s3://<asset>/<user>/starVLA/exp/<RUN_NAME>` every
  `CKPT_MIRROR_INTERVAL=300`s; also on the EFS PVC at
  `/efs/<user>/exp/starvla/<RUN_NAME>`.

Useful knobs: `DRY_RUN=1` (render the job JSON without submitting),
`SYNC_CODE=0`, `NPROC_PER_NODE`, `num_train_epochs`, `data_mix`,
`CUSTOM_JOB_NAME`.

## Eval (sim)

`eval_files/` implements the RMBench policy-module contract
(`get_model` / `eval` / `reset_model`, see `examples/eval_protocol.md`) as a
thin WebSocket client of the starVLA policy server:

```bash
# one task (server is started/stopped automatically)
bash examples/simBenchmarks/RMBench/eval_files/eval_rmbench.sh \
    -c playground/Checkpoints/<run>/checkpoints/steps_X_pytorch_model.pt \
    cover_blocks

# the 9 trained tasks, sequentially, one server lifetime
bash examples/simBenchmarks/RMBench/eval_files/eval_rmbench.sh -c <ckpt.pt> all
```

Prerequisites:

- `RMBENCH_HOME` (default `/code/RMBench`) with the sim env installed
  (`bash script/_install.sh` upstream); set `RMBENCH_PYTHON` to its python.
  That env also needs `pip install websockets` (client transport).
- `STARVLA_PYTHON` (default: conda env `starVLA`) for the policy server.

How it works: `eval_rmbench.sh` symlinks `eval_files/model2rmbench_interface.py`
to `$RMBENCH_HOME/policy/starvla_rmbench.py`, starts
`eval_files/run_policy_server.sh` (checkpoint + `dataset_statistics.json` are
read from the run dir), waits for the port, then runs upstream
`script/eval_policy.py` per task. Results land in
`$RMBENCH_HOME/eval_result/<task>/starvla_rmbench/...` and are copied to
`results/rmbench_eval/<run>/` next to the per-task logs.

Train/test consistency (all verified against the converted data):

- env qpos `[l_arm, l_grip, r_arm, r_grip]` is reordered to the model's
  `state_keys` layout `[l_arm, r_arm, l_grip, r_grip]` and normalized with the
  checkpoint's stats (joints min_max→[-1,1], grippers binary @0.49); predicted
  actions are un-normalized server-side and reordered back.
- images: `[head, left_wrist, right_wrist]` @ 224²; WM history (112²) is
  supported via `history_frames` in `deploy_policy.yml` — keep `0` unless the
  checkpoint was trained with `framework.working_memory`.
- `instruction_type: unseen` is the benchmark default; the converted training
  data used `seen[0]`, so `seen` measures fit and `unseen` measures
  instruction generalization.

`TEST_NUM` is the total number of valid episodes per task (default 100).
The runner overrides the upstream constant in memory; it never edits the shared
checkout to allocate episodes.

### Batched evaluation

```bash
EVAL_TASK=battery_try BATCH_SIZE=4 TEST_NUM=100 BATCH_WAIT_MS=20 \
EVAL_CKPT=s3://your-run/checkpoints/steps_15000_pytorch_model.pt \
bash submit_eval_rmbench.sh
```

`BATCH_SIZE` is both the maximum inference batch and the number of rollout
workers (capped by `TEST_NUM`). It defaults to 1, which uses the serial entry.
The general policy server keeps batching disabled unless `--batch_size > 1`.
`BATCH_WAIT_MS` is a nonnegative integer, default 20: a partial batch runs when
that time has elapsed since its first queued request. Incompatible normalization
keys or inference options run separately. Action chunk lengths are unchanged.

One sequential expert process selects seeds and instructions; workers never
replace a rejected seed. Python instruction randomness is fixed by episode seed
in both modes (upstream previously left this unseeded). The selector repeats the
upstream post-expert setup before choosing the instruction, so it can be a
throughput bottleneck. It holds at most twice the worker count in its job queue.
Each process has its own output directory, simulator, client history and action
cache. Results and videos retain global episode IDs. `episodes.jsonl` is sorted
by ID; `selector/episodes.jsonl` also preserves expert information. Worker or
inference errors abort the task, write diagnostics and clean up children; they
are not counted as unsuccessful episodes. OOM never silently reduces the batch.

Run the same checkpoint with `BATCH_SIZE=1`, `2`, and `4` on H200 to compare
`inference batch_size=... inference_ms=...`, `episodes_per_second`, and the CUDA
peak allocated/reserved byte logs. Check that batch sizes above one actually
occur. Compare deterministic action outputs on identical observations with a
numerical tolerance; stochastic predictions need not match bit for bit. This
hardware and checkpoint validation is required before claiming a speedup or
numerical equivalence. Success-rate improvement is not an acceptance criterion.

### Cluster submission (KOALA)

For the episode-anchor + four-recent-frame WM recipe, use
[ANCHOR_MEMORY.md](ANCHOR_MEMORY.md). It includes the paired training/eval
commands, checkpoint-driven input layout, real processor sizes and profiling.

To compare checkpoint predictions against converted training demonstrations,
submit an open-loop replay job. It stages the task's existing converted data
from `datasets/rmbench_lerobot/<task>` on S3, starts the same policy server,
and uploads per-step action errors, predicted/reference chunks, and images.
It skips simulator setup and reports action error, not success rate:

```bash
REPLAY_TRAINING=1 REPLAY_EPISODES="0 1 2" REPLAY_STEPS="0 35 50 70 100 148" \
BATCH_SIZE=1 SYNC_CODE=1 EVAL_TASKS=observe_and_pickup \
EVAL_CKPT=s3://your-run/checkpoints/steps_26122_pytorch_model.pt \
bash submit_eval_rmbench.sh
```

Results are under `logs/*/training_replay/<task>/episode_*/` in the usual
S3 eval output directory, including `metrics.json`, `replay.log`, and NPZ/PNG
files. Episode and step indices are zero-based and must exist in the dataset.
Replay reconstructs recent history and the episode anchor when the checkpoint
requires them; no-WM checkpoints retain their original three-view input.

For failed rollouts, `TRACE_INFERENCE=1` records the exact three-camera
observations, normalized state and predicted actions at every inference, plus
per-step TOPP outcomes and task `fail_flag` transitions. This works with both
`eval_rmbench.sh` and `submit_eval_rmbench.sh`; it defaults off. Records are
written to `<task-log-run>/diagnostics/<task>/episode_NNNNNN/` and included in
the normal result upload:

```bash
TRACE_INFERENCE=1 EVAL_TASK=observe_and_pickup TEST_NUM=2 \
EVAL_CKPT=s3://your-run/checkpoints/steps_26122_pytorch_model.pt \
bash submit_eval_rmbench.sh
```

`inference_*.npz` contains `raw_image_0..2`, `model_image_0..2` (head, left,
right), `state_env`, `state_input`, `actions_model`, `actions_env` and
`instruction`. `execution.jsonl` includes TOPP exceptions even when upstream
catches them. A TOPP fallback on an unchanged arm can be expected; use
`max_joint_delta` to distinguish it from a requested move that was not
executed. Diagnostics preserve upstream fallback and success conditions.
For a directly loaded policy module, set `trace_dir` in `deploy_policy.yml`
or `RMBENCH_TRACE_DIR` in the simulator process instead.

`submit_eval_rmbench.sh` (repo root) runs the same flow in a pod: the
`:rmbench-eval` image (`deployment/docker/Dockerfile.rmbench_eval` — training
image + a `rmbench` conda env with sapien 3.0 / curobo / pytorch3d, built once
with `BUILD_IMAGE=1`), RMBench source+assets synced local → S3 → EFS (cached
across pods), then `eval_files/run_rmbench_eval_in_pod.sh` loops over tasks —
one checkpoint + one policy server per task.

```bash
# first-time setup: build the image and upload RMBench (code + assets)
BUILD_IMAGE=1 SYNC_RMBENCH=1 bash submit_eval_rmbench.sh

# regular eval of the 9 tasks against a training sweep run
TRAIN_RUN_PREFIX=/efs/danyangchen/exp/starvla/train-task1-20260912_022536 \
    bash submit_eval_rmbench.sh

# quick smoke: one task, 2 episodes, explicit checkpoint
EVAL_TASKS=cover_blocks TEST_NUM=2 \
EVAL_CKPT=/efs/.../steps_X_pytorch_model.pt \
    bash submit_eval_rmbench.sh
```

Per-task checkpoints are resolved as the latest
`${TRAIN_RUN_PREFIX}_<task>/checkpoints/steps_*_pytorch_model.pt` (the
`submit_train_rmbench.sh` sweep naming). `TRAIN_RUN_PREFIX`/`EVAL_CKPT` accept
both EFS paths and `s3://` URIs — the train job mirrors every run dir
(`config.yaml` + `dataset_statistics.json` + `checkpoints/`) to
`s3://<asset>/<user>/starVLA/exp/`, and the pod stages S3 checkpoints onto EFS
before eval, so evaluating a checkpoint that only exists on S3 works:

```bash
TRAIN_RUN_PREFIX=s3://helix-asset-ap-northeast-1/danyangchen/starVLA/exp/train-task1-20260912_014344 \
    bash submit_eval_rmbench.sh
```

Results are printed to the pod log three ways: live `Success!/Fail!` lines per
episode, a per-task summary when each task finishes, and a final consolidated
dump of every `_result.txt` plus a success-rate table at the end of the job.
They also land in `$RMBENCH_EFS_DIR/eval_result/` + `EVAL_OUTPUT_DIR` on EFS
and are synced to
`s3://<asset>/<user>/starVLA/exp/rmbench-eval/<RUN_NAME>`; pull locally with
`aws s3 sync <that prefix>/ ./evaluate_results/<RUN_NAME>/`.

## Follow-ups

- h16 variant (`robotwin` data config + `action_horizon: 16`)
- q99 normalization variant (RoboDojo-style)
- VLM co-training (add back a `datasets.vlm_data` block, Robotwin-style)
# 从 checkpoint 恢复训练（KOALA）

`submit_train_rmbench.sh` 使用 `RESUME_CKPT` 指定恢复来源，同时需通过
`TRAIN_TASK`（或只含一个任务的 `TRAIN_TASKS`）选择对应任务：

```bash
# 提交机本地目录：提交时自动上传到 S3 asset，即使 SYNC_ASSET=0 也会上传
TRAIN_TASK=battery_try RESUME_CKPT=/data/checkpoints/old_battery_try bash submit_train_rmbench.sh

# S3 run 目录或 checkpoints/ 前缀：pod 内通过 AWS CLI 下载
TRAIN_TASK=battery_try RESUME_CKPT=s3://bucket/user/starVLA/exp/old_battery_try bash submit_train_rmbench.sh

# EFS：使用 pod 中的 /efs/... 路径，直接读取，不要求提交机挂载该目录
TRAIN_TASK=battery_try RESUME_CKPT=/efs/user/exp/starvla/old_battery_try bash submit_train_rmbench.sh
```

支持 run 目录、`checkpoints/` 目录及具体的 `steps_N_model.safetensors` /
`steps_N_pytorch_model.pt` 文件。目录自动选步数最大的文件，同一步数优先
safetensors；缺少有效 checkpoint 会报错，只有 `final_model/` 无法恢复步数。

恢复模型权重和训练步数，并据此调整学习率；当前训练器没有保存 optimizer、
随机数和 dataloader 状态，因此不是逐位一致的续训。`num_train_epochs` 是包含
已完成步数的总训练预算。应保持原任务和训练配置一致，并使用新的 `RUN_NAME`
保存续训结果；EFS 输入位于新输出目录内时会拒绝启动，以免被输出清理逻辑删除。
`DRY_RUN=1` 沿用提交脚本语义：仅跳过 KOALA 提交，仍可能执行上传。
