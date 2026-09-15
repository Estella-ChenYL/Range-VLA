# Working Memory 重设计：历史帧直拼多图输入 + M-RoPE 时间轴

日期：2026-09-08
状态：已批准（路线 A）
替换了：2026-09-07 回退的 cross-attention resampler 设计（备份于 `/tmp/wm_design_backup_20260907.patch`）

## 背景与动机

旧设计用 cross-attention resampler（HistoryTokenizer）把 4 帧历史压成 8 个可学习 token 注入 VLM
前缀，配套 null_token / pad_embed / history_dropout 三套条件分支与独立 LR 组。该设计在训练中
出现 `null_token` 参数被 NaN 静默污染的事故：条件分支参数 + 冻结 VLM 反向梯度尖峰 + ZeRO-1/2 下
gradient guard 对 dict 结构的 `averaged_gradients` 失效，三者叠加导致参数与 Adam 状态永久损坏。

新设计的核心原则：**结构上消灭故障类别，而不是加防御层**。不引入任何新参数、新模块、条件分支。

## 设计决策（已与用户确认）

| 决策点 | 结论 |
|---|---|
| 形式 | 4 个历史帧直接作为图片拼进 VLM 多图输入 |
| 视角 | 历史只用 primary；当前帧保持 primary + wrist |
| 时间编码 | 多图输入 + 自定义 M-RoPE T 轴（等间距） |
| 推理效率 | 历史帧降分辨率（112 vs 当前帧 224） |
| 帧间隔 | 带 stride，deltas `[-8,-6,-4,-2,0]`（默认 stride=2） |
| episode 开头 | 复制当前观测（dataloader clamp 原生语义） |
| 范围 | 训练 + LIBERO 在线评测全链路 |

## 架构

```
训练:
  dataloader (primary 取 deltas [-8,-6,-4,-2,0], wrist 取 [0])
    → example["image"] = [hist0, hist1, hist2, hist3, primary_t, wrist_t]
      (历史帧 112×112, 当前帧 224×224; episode 开头 clamp 到帧 0 = 复制当前观测)
    → build_qwenvl_inputs: content 按序塞 6 张图 + 文本
    → position_ids 后处理: T 通道按帧等间距赋值, H/W 不动
    → 冻结的 Qwen3-VL → hidden states → action DiT（完全不变）

评测:
  model2libero_interface 维护 primary 帧 deque（每个 env step 入队）
    → 查询时按同样 stride 取 [t-8, t-6, t-4, t-2]，不足则复制当前帧
    → 拼成与训练逐位一致的 6 图布局，走同一个 build_qwenvl_inputs
```

### 布局契约

`example["image"]` 的顺序 = `[4 张历史 primary（旧→新）, 当前 primary, 当前 wrist]`。
该顺序同时决定：content 里图的顺序 → `image_grid_thw` 的顺序 → T 轴赋值的顺序。
`build_qwenvl_inputs` 在 working_memory 启用时 assert 图数 == `history_frames + 当前视角数`，
不符合直接报错（fail loud，不静默退化）。

### 配置

```yaml
framework:
  working_memory:
    history_frames: 4        # 历史帧数（不含当前帧）
    history_stride: 2        # 帧间隔 → deltas [-8,-6,-4,-2,0]，覆盖 8 个 env step
    history_image_size: 112  # 历史帧分辨率（当前帧 224）
    mrope_t_stride: 2        # T 轴相邻帧间隔
```

不配这一节 = 行为与 HEAD 逐位一致（回归安全）。

### token 成本

224 图 = 64 token（8×8），112 图 = 16 token（4×4）。
视觉 token：4×16 + 64 + 64 = **192**，对比 baseline 128，+50%。

## M-RoPE T 轴自定义

Qwen3-VL 默认行为（transformers 5.12.1 `get_rope_index`）：每张图的 T 位置 = 前一模态结束位置，
图间 T 间距 ≈ 前图宽度（patch 数），即"时间间隔"与分辨率耦合且不可控。

本设计在 `build_qwenvl_inputs` 内新增 `_apply_history_mrope(batch_inputs, wm_cfg)`：

1. 用模型自带的 `get_rope_index`（位于内层 `Qwen3VLModel`，即 `self.model.model`）算出默认
   `position_ids`（3, B, L）
2. 只覆写 T 通道：6 张图的 T 分别赋为 `[0, s, 2s, 3s, 4s, 4s]`（s = mrope_t_stride，
   wrist 与当前 primary 同时刻）
3. H/W 通道、文本位置不动。帧最大 T = 8 < 文本起始 T（≈32，由图宽累加），单调性天然保持
4. 将修改后的 `position_ids` 传入 `self.model(...)`（HF 前向在显式传入时不重算）

### 关于未来的 Incident Memory（已决策：不预留 T 槽）

考虑过给将来的 incident 触发式检索记忆预留 T 轴 headroom（`mrope_t_offset`），
**决定不预留**：WM-v1 保持 `[0,2,4,6,8,8]`。Incident Memory 引入时，再根据实际的检索条数
和 memory token 形式动态生成整个 multimodal position layout，现阶段不为未确定的 memory
形式固定 T 槽。（注意：position_ids 不可用负数，检索记忆只能放在历史帧之前的低位区间，
届时通过整体上移历史帧 T 值实现。）

## 组件改动清单

### 1. `starVLA/dataloader/lerobot_datasets.py` — 数据集工厂

- 工厂在构建 modality_config 后、当 `history_frames > 0` 时，把 video modality 的 delta_indices
  覆写为由 `history_frames`/`history_stride` 生成的历史 deltas（`[-8,-6,-4,-2,0]`）。
  该覆写对所有 video key 生效（wrist 也会采样 5 帧，打包时只取最后一帧 = 当前帧；
  接受 wrist 多解码 4 帧的开销以换取最简实现）
- `wm_cfg` 经 build_dataloader → get_vla_dataset → make_LeRobotSingleDataset 传递，
  最终挂在 `dataset.wm_cfg` 上
- `history_frames=0` 时 deltas 保持 `[0]`，与现状一致

### 2. `starVLA/dataloader/gr00t_lerobot/datasets.py` — `_pack_sample`

- 打包委托给 `working_memory.pack_step_images`：history key（primary）取全部帧（旧→新），
  历史帧 resize 到 `history_image_size`（BILINEAR，与评测侧一致）；wrist 取最后一帧 resize 224
- 边界复制由 `get_video` 的 `np.maximum(step_indices, 0)` clamp 天然保证，无需新代码

### 3. `starVLA/model/modules/vlm/QWen3.py` — `build_qwenvl_inputs`

- 新增 `_apply_history_mrope`（约 40 行），见上节
- 唯一碰模型输入的改动

### 4. `examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py`

- 现有 `image_history` deque 改为存 primary 帧（每个 env step 入队）
- 查询时按 stride 取 4 帧、不足复制当前帧、resize 到 `history_image_size`，
  拼在 `example["image"]` 最前面
- 与训练布局逐位一致（延伸该文件已有的 "TRAIN/TEST CONSISTENCY" 约定）
- `history_frames` 在评测入口（eval_libero.py Args）默认 **0**（opt-in）：
  评测无 WM 的 checkpoint 时绝不能静默发 6 张图；评测 WM checkpoint 时显式设 4

## 数值稳定性

- **零新参数**：无 resampler / null_token / pad_embed，不存在"条件分支决定是否收到梯度"的参数
- **梯度路径与已知稳定 baseline 完全相同**：VLM 冻结，可训的只有 action_model / projector
- **bf16 前向无新算子**：历史帧只是更多 patch token
- 教训记录：若将来重新加 gradient guard，ZeRO-1/2 的 `averaged_gradients` 是
  `dict[int, list[Tensor]]`，必须遍历 `.values()`，直接遍历 dict 得到的是整数 key

## 测试

| 层 | 内容 |
|---|---|
| 单元（CPU） | dataloader：6 张图、顺序、episode 开头复制当前帧、尺寸 112/224 正确 |
| 单元（CPU） | T 轴：构造假 `input_ids` + `image_grid_thw`，验证 `position_ids[0]` 在 6 个图区间分别为 `[0,2,4,6,8,8]`，H/W 通道与默认一致 |
| 冒烟 | 单步 forward+backward，loss 有限；`working_memory` 缺省时与 HEAD 输出逐位一致 |
| 验收 | 40+ 步训练无 NaN；LIBERO 评测与无历史 baseline 对比成功率 |

## 明确不做（YAGNI）

- 文本时间戳标记（如 `<frame t-3>`）
- KV cache 跨步复用（滑动窗口重叠帧）
- 历史帧专用 attention 掩码
- 可学习时间嵌入

以上均需 baseline 对比证明有价值后再考虑。
