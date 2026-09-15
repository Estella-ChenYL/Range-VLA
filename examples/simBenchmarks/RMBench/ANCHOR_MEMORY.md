# RMBench episode anchor + four-frame WM

This recipe uses an episode anchor in addition to the existing four recent
head-camera frames. It does not implement incident candidates, an FSM, retrieval,
or a new DiT fusion mechanism. Generic `working_memory.py` is unchanged; the
dataset and VLM factories dispatch to RMBench adapters only for this recipe.

| Role | Image index | Source observation index at step t | Input size | mRoPE T |
|---|---:|---|---|---:|
| episode anchor (head) | 0 | 0 | 112×112 | 0 |
| recent head 0 | 1 | max(0, t−8) | 112×112 | 2 |
| recent head 1 | 2 | max(0, t−6) | 112×112 | 4 |
| recent head 2 | 3 | max(0, t−4) | 112×112 | 6 |
| recent head 3 | 4 | max(0, t−2) | 112×112 | 8 |
| current head/left wrist/right wrist | 5/6/7 | t | 224×224 | 10/10/10 |

T denotes input order, not elapsed time. Frame indices are separately retained
in `rmbench_memory.frame_ids`. The anchor is loaded from the same episode even
when training samples its middle. Evaluation caches the first visible head
frame, updates recent memory on every observation, and clears both on episode
reset. A language change inside an episode does not erase its anchor.

Each image has a separate role and token span (`start`, `end`, end-exclusive).
Recent WM corresponds to image indices 1–4, never 0–4. No future incident
candidate implementation should count the anchor as newly observed evidence.

## Actual processor dimensions and cost

With the locally installed Qwen3-VL processor (patch 16, merge 2), its original
minimum area promotes both 112 and 224 inputs to 256×256. The RMBench adapter
uses a compact processor for anchor/history only: 112 inputs align to 128×128,
16 merged visual tokens each. Current views use the original processor and
remain 256×256, 64 tokens each; their processed tensors are checked unchanged.
Total: 272 visual tokens/sample, including 16 anchor tokens. This excludes
text and image delimiter tokens. Actual grids are recorded, not assumed.

This compact history preprocessing is part of this recipe. A later seven-image
WM ablation must use the SAME compact history preprocessing: the legacy WM
processor would otherwise produce 448 visual tokens for seven images, making
the comparison confounded. No seven-image ablation is claimed to have run.

The VLM logs `[RMBENCH_ANCHOR_LAYOUT]` once and
`[RMBENCH_ANCHOR_PROFILE]` on the first/every 100th VLM forward. The latter
reports VLM forward latency and process-wide peak allocated/reserved GPU memory.
Training metrics additionally report microstep time including backward and the
optimizer. Policy-server batching logs report full inference time/GPU peaks;
the client logs inference round-trip latency. First-call warmup and batch size
must be considered when comparing these numbers. GPU cost has not been measured
in this CPU workspace.

## Koala training and evaluation

Start a NEW run from the base VLM. This is not an inference-only extra image
on the old no-WM checkpoint. Keep `ANCHOR_RUN` for the subsequent eval command:

```bash
cd /code/starVLA
export ANCHOR_RUN="rmbench-anchor-wm4-$(date +%Y%m%d_%H%M%S)"

RUN_NAME="$ANCHOR_RUN" CUSTOM_JOB_NAME=rmbench-anchor-wm4 \
TRAIN_TASKS=observe_and_pickup \
config_yaml=examples/simBenchmarks/RMBench/train_files/starvla_qwenpiv3_rmbench_anchor.yaml \
NPROC_PER_NODE=6 per_device_bs=24 num_train_epochs=500 \
RESUME_CKPT= SYNC_CODE=1 SYNC_RAW=0 SYNC_ASSET=0 \
bash submit_train_rmbench.sh
```

These batch settings retain the earlier effective global batch of 144 and
26,122 optimizer steps for 7,523 frames over 500 epochs. They have NOT been
GPU-memory qualified for this eight-image recipe. If memory is insufficient,
reduce the microbatch deliberately and record the resulting optimizer-step
budget; do not silently drop the anchor or a WM frame.

After training finishes, evaluate its latest saved checkpoint:

```bash
TRAIN_RUN_PREFIX="s3://helix-asset-ap-northeast-1/danyangchen/starVLA/exp/${ANCHOR_RUN}" \
EVAL_CKPT= EVAL_TASKS=observe_and_pickup \
BATCH_SIZE=1 TEST_NUM=20 TRACE_INFERENCE=1 \
SYNC_CODE=1 SYNC_RMBENCH=0 REPLAY_TRAINING=0 \
bash submit_eval_rmbench.sh
```

The eval submitter appends `_observe_and_pickup` to this run prefix. If using
another terminal, set `ANCHOR_RUN` to the run name printed by training submission.
The policy-server handshake carries the checkpoint's memory contract; the client
auto-selects the layout and rejects conflicting overrides. Diagnostic NPZs include
frame IDs, image roles, actual processor token spans, and inference latency.

For open-loop replay of this new checkpoint use the same eval command with
`REPLAY_TRAINING=1 REPLAY_EPISODES="0 1 2" REPLAY_STEPS="0 35 50 70 100 148"`.
Replay reconstructs the anchor and history from the complete demonstration
prefix before each sampled query.

An explicit seven-image fallback is available by making a separate recipe
with `working_memory.history_frames: 3`. Use it only after a measured resource
or interface limit, train it separately, and label it as a different experiment.
No automatic fallback is enabled.

## Verification

`tests/test_rmbench_anchor.py` covers long episodes, mutable camera buffers,
reset and environment isolation, missing observations, checkpoint auto-configuration,
and conflicting overrides. Existing generic WM tests remain applicable.

`tests/test_rmbench_anchor_integration.py` uses local RMBench data and the local
Qwen processor in the training Python environment. It compares real training
and eval image/state tensors across episodes, verifies current-camera processing
is unchanged, and runs eight-image mRoPE inputs through a tiny random Qwen3-VL
on CPU. It does not establish full-checkpoint GPU training stability or success rate.
