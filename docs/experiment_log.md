# 实验日志

## 2026-09-14：环境、渲染与首批数据

| 项目 | 已验证结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090（32 GB），PyTorch CUDA 可用 |
| 仿真 | ManiSkill 3.0.1 / SAPIEN 3.0.3 / PhysX CUDA |
| Vulkan | 默认 GLX ICD 只识别 llvmpipe；项目内 EGL ICD 配置后识别 RTX 5090 |
| RGB 冒烟测试 | PushCube-v1，4 并行环境，20 steps，80 transitions，177.1 FPS |
| 视觉观测 | `base_camera.rgb`，`(N, 128, 128, 3)`，`uint8`，GPU tensor |
| 首批数据 | 官方 motion-planning 示范回放前 10 条，10/10 成功，约 3.9 MB HDF5 |

### Baseline 管线验收（不可作为最终评测）

第一版 CNN encoder/decoder + action-conditioned MLP dynamics 已在首批数据上完成 3 个 epoch 的端到端运行。验证总损失从 `0.06025` 降至 `0.02723`，说明读取、反向传播、checkpoint 和指标落盘路径均正常。

该次运行最初按 transition 随机切分，存在相邻帧泄漏到验证集的风险，因此**不能用于模型优劣结论或简历**。训练脚本现已改为 trajectory-level split；下一次运行的结果才会进入对比表。

### Trajectory-level baseline（初步结果）

使用固定随机种子将 10 条轨迹划为 9 条训练、1 条验证，对应 614 / 72 个转移；输入图像被下采样到 `64 x 64`，latent 维度为 128，训练 3 个 epoch。

| 指标 | Epoch 1 验证 | Epoch 3 验证 |
| --- | ---: | ---: |
| total loss | 0.07683 | 0.04377 |
| next-image MSE | 0.03050 | 0.01428 |
| reconstruction MSE | 0.03048 | 0.01425 |
| latent MSE | 0.000079 | 0.000071 |
| state MSE | 0.01577 | 0.01517 |

解释边界：损失下降证明本轮实现可训练、且对未见完整轨迹有初步泛化；但仅 10 条示范、验证集 1 条轨迹，不能据此声称“达到控制效果”或报告规划成功率。下一轮应扩展数据，并报告均值与方差。

### 100-trajectory baseline（当前可复现结果）

- 回放请求 100 条官方 motion-planning 示范，成功保存 97 条（97.0%）；输出视觉轨迹 HDF5 为 38.1 MB。
- 按完整 trajectory 固定切分为 87 条训练 / 10 条验证，分别为 6,119 / 751 个转移；不会发生同一轨迹帧泄漏。
- CNN autoencoder（128-d latent）+ action-conditioned MLP dynamics，输入 64 x 64 RGB，batch size 64，10 epochs，2 DataLoader workers。

| 验证指标 | Epoch 1 | Epoch 10 |
| --- | ---: | ---: |
| total loss | 0.01952 | 0.01207 |
| next-image MSE | 0.00811 | 0.00491 |
| reconstruction MSE | 0.00809 | 0.00489 |
| latent MSE | 0.000048 | 0.000002 |
| state MSE | 0.00328 | 0.00226 |

结果边界：这是离线、单步预测的 sanity baseline；数值说明训练管线和 held-out trajectory 上的预测损失均稳定下降，但尚未测试多步 rollout、action ablation 或 CEM 闭环控制，不能将上述 MSE 解释成机器人任务成功率。

### 多步 free rollout 与动作消融（当前结果）

使用同一 checkpoint，在 10 条 held-out trajectory 的 596 个起点进行 free latent rollout。每一步把上一步预测的 latent 和 state 再输入 dynamics；`zero_action` 消融仅将输入动作设为零，其余模型权重、起点和评价完全相同。

| Horizon | Action-conditioned pixel MSE | Zero-action pixel MSE | Action-conditioned state MSE | Zero-action state MSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.00477 | 0.00480 | 0.00051 | 0.00125 |
| 5 | 0.00491 | 0.00519 | 0.00345 | 0.01467 |
| 10 | 0.00508 | 0.00580 | 0.00652 | 0.03778 |

在 10 步时，保留动作条件使状态 MSE 相比 zero-action 低约 82.7%，像素 MSE 低约 12.3%。这支持“模型会使用 action 来预测状态转移”的判断；但仍是离线 prediction 证据，下一阶段必须通过 CEM 闭环 success rate 验证控制收益。

### CEM 闭环调试（负结果，保留）

使用 learned dynamics 进行 CEM 搜索，并在真实 PushCube 环境以 `pd_joint_pos` 闭环执行。初版错误地将动作裁剪为 `[-1, 1]`；修复为从 replay HDF5 提取的示范动作均值、方差和边界后，seed 42 的单回合仍未成功：环境 50 步截断，cube-goal 初始/最终二维距离均约 `0.200`。

结论：目前的单步 CNN/MLP 模型可用于动作条件预测消融，但不足以做有效长程 CEM 控制。后续改进顺序是：先扩增至完整 1,000 条示范；训练 multi-step rollout loss；以 behavior-cloning policy 产生 CEM warm start；再在相同 seeds 下报告 CEM、BC 和 random 的 success rate / final distance 对照。

### 已确认数据契约

训练样本是严格的一步转移：`(o_t, s_t, a_t) -> (o_{t+1}, s_{t+1})`。

- `rgb`: `(T+1, 128, 128, 3)`，uint8；
- `state`: `(T+1, 35)`，float32；
- `actions`: `(T, 8)`，float32；
- `rewards` / `success`: `(T,)`。

### 已解决问题

1. **Vulkan 仅识别 llvmpipe。** 原因是无桌面云容器的默认 `libGLX_nvidia.so.0` ICD 初始化失败。解决：用 `configs/nvidia_icd_egl.json` 与 `VK_ICD_FILENAMES` 显式选用 `libEGL_nvidia.so.0`；不修改系统驱动。
2. **官方示范下载不可达。** 直连 Hugging Face 报 `Network is unreachable`。解决：经 `hf-mirror.com` 下载相同的官方 ZIP；记录原始 URL 和 SHA/版本后再做大规模数据下载。
3. **回放后端名称漂移。** CLI help 仍列 `physx_gpu`，当前 ManiSkill 3.0.1 实际接受 `physx_cuda`。本项目统一使用后者。

### 下一步

1. 运行 `scripts/train_dynamics.py`，先验证一阶视觉-状态潜空间动力学损失下降；
2. 扩增回放示范并按 trajectory 划分训练/验证集；
3. 增加 5/10 步 latent rollout drift 与像素预测指标；
4. 在 learned dynamics 上加入 CEM，并报告任务 success rate。

### 控制基线预检（100 条轨迹，2026-09-14）

- 为区分“数据/动作定义问题”和“世界模型规划问题”，新增 `state -> pd_joint_pos action` 的 MLP 行为克隆（BC）基线。它读取模拟器完整状态（包含方块、目标和 TCP 位姿），因此只作为控制可学习性的诊断，**不是**视觉世界模型结果。
- 按完整轨迹拆分：87 条训练、10 条验证，6,119 / 751 transitions；第 10 epoch 验证 raw action MSE 为 `0.0019761`。
- 但在真实闭环的 5 个固定种子回合（42--46）中，成功率为 `0/5`，平均最终 cube-goal 距离为 `0.200004 m`，几乎没有推动方块。
- 结论：当前 100 条数据的一步 MLP-BC 仍不能恢复接触操作中的时序策略；不能用离线动作 MSE 替代真实控制成功率。全量数据完成后将改用 action chunk / history-conditioned policy，并与 BC、CEM 统一对比。

### 全量数据与多步世界模型（975 条成功回放）

- 请求回放 1,000 条官方 motion-planning 示范，成功保存 975 条（97.5%）。按整条轨迹固定切分为训练 / held-out trajectory，避免相邻帧数据泄漏。
- 单步 CNN autoencoder（128-d latent）+ action-conditioned MLP dynamics 训练 20 epochs：held-out `next-image MSE = 0.000832`、`state MSE = 0.002823`。
- 以上 checkpoint 初始化 5-step free-rollout 微调后，在 98 条 held-out trajectories、5,888 个起点上复测：

| Horizon | Action-conditioned pixel MSE | Zero-action pixel MSE | Action-conditioned state MSE | Zero-action state MSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.000636 | 0.005035 | 0.000662 | 0.016723 |
| 5 | 0.000821 | 0.012171 | 0.002776 | 0.300130 |
| 10 | 0.001270 | 0.030323 | 0.005832 | 5.358248 |

- 解释边界：10-step conditioned state MSE 明显低于 zero-action，说明模型没有忽略 action；这仍是**离线预测**结果，不能等价为控制成功。
- 原始 terminal-distance CEM 在 10 个固定种子中成功率 `0/10`，平均最终 cube-goal 距离 `0.227221 m`；对齐环境分阶段稠密奖励后的 3 回合调试也为 `0/3`，其中 1 回合出现模型利用导致的方块远离目标。结论是当前 deterministic dynamics + unconstrained CEM 尚不适合报告为可用控制器。
- 全量 raw-action BC（50 epochs）在 held-out action MSE 上达到 `0.0004321`，但真实闭环 20 个种子仍为 `0/20`、平均最终距离 `0.199994 m`。这再次证明不能以离线 action MSE 替代任务 success；现正评测 joint-delta 表示，并准备使用 history / action-chunk 形式改善接触控制。

### 评测协议修正与统一 100-step 对照

- 发现官方 motion-planning 示范长度为 61--151 steps（中位数 69），而 `PushCube-v1` 的 Gym 注册默认 `max_episode_steps=50`。先前 BC/CEM/kNN 闭环评测实际在第 50 步被截断，因此其中的 0% success **不能作为控制失败结论**。
- 所有后续控制评测显式传入 `max_episode_steps=100`，并以固定 seeds 42--61 重跑。100-step 上限只用于与官方示范时长匹配；所有报告都会明确标示该协议。

| 方法 | 观测 | 回合数 | 成功率 | 平均最终 cube-goal 距离 |
| --- | --- | ---: | ---: | ---: |
| kNN action-chunk retrieval（k=1, chunk=2） | simulator privileged state | 20 | **70%** | **0.120881 m** |
| raw-action MLP BC | simulator privileged state | 20 | 0% | 0.199994 m |
| joint-delta MLP BC | simulator privileged state | 20 | 0% | 0.474117 m |
| terminal-cost CEM + learned visual dynamics | RGB + state | 10 | 0% | 0.329589 m |

- kNN action-chunk 的 70% 表明：专家数据覆盖、状态接口和 100-step 控制协议均可工作。对照中，逐步检索为 60%，执行 2 步连续专家动作后再检索提升至 70%；它是非参数、特权状态的**诊断上界/基线**，不可表述为视觉世界模型成功率。
- CEM 仍有 model exploitation，下一阶段应使用动作序列检索/BC warm-start、uncertainty penalty 或 ensemble dynamics，而不是扩大无约束的动作采样范围。
