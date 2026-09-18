# Cosmos Policy + LIBERO 闭环评测

## 目标

在 NVIDIA 公开的 `Cosmos-Policy-LIBERO-Predict2-2B` 检查点上，完成 LIBERO 四个标准任务套件的可复现闭环评测，并在同一评测协议下分析动作分块和视觉扰动对成功率的影响。

本实验使用预训练策略做推理与评测；不将其表述为在单卡上完成的全量训练。

## 第一阶段：官方复现

- 官方源码：`NVlabs/cosmos-policy`。
- 基准：LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-10。
- 检查点：`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`。
- 产物：每个任务套件的逐任务成功率、总成功率、固定随机种子、代表性闭环视频。

### 已完成：LIBERO-Spatial smoke run（2026-09-16）

这是一轮验证整条评测链路的最小实验，不是官方论文级的全量复现。

| 项目 | 实际配置 / 结果 |
| --- | --- |
| 代码与权重 | `NVlabs/cosmos-policy`；`nvidia/Cosmos-Policy-LIBERO-Predict2-2B` |
| 硬件 | NVIDIA RTX PRO 6000 Blackwell Server Edition（约 96 GB VRAM） |
| 任务 | `libero_spatial`，10 个任务，每任务 1 条轨迹 |
| 随机性 | policy seed `195`；`deterministic=True`；reset seed `0` |
| 策略配置 | wrist image + agent-view image + proprioception；action chunk `16`；open-loop steps `16`；5 次 action denoising |
| 闭环结果 | `10 / 10` success，point estimate `100.0%` |
| 不确定性 | 10 条样本的 95% Wilson 区间约为 `72.2%--100%`；样本量过小，不能与官方多 seed、每任务 50 条的结果直接比较 |
| 性能观测 | 采样到的两次 action query 分别为 `0.442 s`、`0.428 s`；运行时显存约 `6.6 GB` |

评测过程保存了每条 rollout 的 MP4 以及带 future-image 输出的 MP4。视频、权重和 LIBERO 资产均留在云端，不纳入 Git。

### 已完成：LIBERO-Spatial 单种子基线（2026-09-16）

在 smoke run 验证闭环链路后，使用完全相同的模型、seed、环境和动作参数运行每任务 5 条轨迹：

| 项目 | 实际结果 |
| --- | --- |
| 任务数 / 轨迹数 | 10 个 Spatial 任务，`5` trials / task，合计 `50` 条 |
| 成功数 / 成功率 | `50 / 50`，`100.0%` |
| 95% Wilson 区间 | `92.9%--100%` |
| 可比性边界 | 单一 policy seed、单一 deterministic reset seed；官方完整协议为每任务 50 条且建议跨 seed 汇总，本结果不能替代该协议 |

本轮全程采用闭环执行：每 16 个 simulator steps 重新查询一次 action chunk，未使用 demonstration replay 或特权状态控制。原始控制台日志、逐回合 success 记录和 MP4 已保存在云端的 `runs/` 与 `rollouts/` 目录。

### 已完成：四套标准 LIBERO 套件的单种子闭环评测（2026-09-18）

在相同的预训练 checkpoint、双 RGB 观测（agent-view + wrist）、proprioception、policy seed `195`、deterministic reset seed `0` 与 action chunk / open-loop horizon `16` 下，对四个官方任务套件均运行每任务 5 条真实闭环轨迹：策略输出 action 后由仿真器执行，下一次决策读取新的图像和机器人状态；不使用 demonstration replay 或环境特权状态控制。

| 标准套件 | 任务数 | 轨迹数 | 成功数 / 成功率 | 95% Wilson 区间 |
| --- | ---: | ---: | --- | --- |
| LIBERO-Spatial | 10 | 50 | `50 / 50 = 100.0%` | `92.9%--100%` |
| LIBERO-Object | 10 | 50 | `50 / 50 = 100.0%` | `92.9%--100%` |
| LIBERO-Goal | 10 | 50 | `50 / 50 = 100.0%` | `92.9%--100%` |
| LIBERO-10 | 10 | 50 | `47 / 50 = 94.0%` | `83.8%--97.9%` |
| 合计（描述性汇总） | 40 | 200 | `197 / 200 = 98.5%` | `95.6%--99.5%` |

LIBERO-10 的 8 个任务为 `5/5`；两个更长程、多阶段任务分别为 `4/5` 与 `3/5`。这说明该 checkpoint 在本固定初始状态协议下并非所有任务都饱和。上述 `98.5%` 是 200 条样本的描述性合计，**不是**官方完整基准分数：官方通常以每任务 50 条、多个 seed 的汇总作为可比较结果。原始日志分别为云端 `runs/libero_spatial_baseline_h16_n5_reboot.log`、`runs/libero_object_baseline_h16_n5.log`、`runs/libero_goal_baseline_h16_n5.log` 和 `runs/libero10_baseline_h16_n5.log`。

### 已完成：固定亮度域偏移（2026-09-16）

在 horizon `16` 基线上，将 agent-view 与 wrist 两路 RGB 像素同时乘以 `0.6`（40% 全局变暗），不修改 proprioception、初始状态、动作配置或 policy seed。50 条完整评测得到 `50/50 = 100.0%`（95% Wilson `92.9%--100%`）。因此该确定性中等亮度变化没有造成可观测退化；它是视觉鲁棒性对照，而不是只收集失败样例。

### 已完成：中心遮挡域偏移（2026-09-16）

继续以 horizon `16` 基线为对照，在 agent-view 与 wrist 两路 RGB 的几何中心各施加边长为画面 `30%` 的黑色方形遮挡（约覆盖单张图像 `9%` 面积）。除该输入像素变换外，不修改 proprioception、动作配置、初始状态或 policy seed。50 条完整评测得到 `47/50 = 94.0%`（95% Wilson `83.8%--97.9%`）。

与亮度 `0.6` 条件的 `50/50` 相比，双视角均有局部视觉缺失时出现了 3 条失败轨迹，但策略仍保留较高成功率。结合 horizon `16 → 12 → 8` 的急剧退化，这个受控对照表明：在当前协议内，策略对执行时序/动作分块设定的敏感性显著强于对这一级别中心遮挡的敏感性；这不是对所有遮挡类型或所有任务的泛化结论。

### 已完成：强中心遮挡失效边界（2026-09-16）

在完全相同的 horizon `16` 评测协议下，将两路 RGB 的中心黑色方形遮挡边长提高到画面 `50%`（约覆盖单张图像 `25%` 面积）。50 条完整评测得到 `8/50 = 16.0%`（95% Wilson `7.8%--29.2%`）。与 `30%` 边长遮挡的 `47/50 = 94.0%` 相比，遮挡面积从约 `9%` 提高至约 `25%` 后成功率陡降 `78` 个百分点。

因此，当前实验可报告一个明确的受控现象：该预训练策略能容忍双视角中等尺度中心缺失，但在更大面积的同时遮挡下显著失效。该结论基于 `LIBERO-Spatial`、固定 policy/reset seed 及每任务 5 个默认初始状态；后续应增加随机遮挡位置、相机单独遮挡与跨 seed 评测来检验其稳健性。

### 已完成：执行 horizon 敏感性探索（2026-09-16）

保持 checkpoint、任务套件、policy seed、deterministic reset 和 `chunk_size=16` 不变，仅把 `num_open_loop_steps` 从 `16` 改为 `4`。这意味着策略仍预测 16 步 action chunk，但每执行 4 步便重新观测和查询策略。

| 条件 | 完成轨迹 | 成功率 | 95% Wilson 区间 | 说明 |
| --- | ---: | ---: | --- | --- |
| horizon `16`（基线） | 50 | `50/50 = 100.0%` | `92.9%--100%` | 每任务 5 条完整基线 |
| horizon `12` | 50 | `18/50 = 36.0%` | `24.1%--49.9%` | 每任务 5 条完整评测；出现明显的中间性能区 |
| horizon `8` | 50 | `1/50 = 2.0%` | `0.35%--10.5%` | 每任务 5 条完整评测；唯一成功轨迹保留为 failure-boundary 案例 |
| horizon `4`（探索性 pilot） | 14 | `0/14 = 0.0%` | `0%--21.5%` | 已完成前两个任务的 10 条及第三个任务的 4 条；因连续失败且云端 GPU 计费，未继续到原计划 50 条 |

各条件单次 action query 的实测时延均约 `0.42--0.45 s`。因此缩短 horizon 虽增加观测反馈频率，也会显著增加模型调用次数；在此 checkpoint 的训练/推理设定下，成功率沿 `16 → 12 → 8` 呈现 `100% → 36% → 2%` 的陡峭退化，`4` 步重规划在探索性样本上完全失败。该结果是有限样本的诊断证据，不泛化为所有任务或所有策略的结论。

### 环境与兼容性记录

- 无头渲染使用 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl` 和 `MUJOCO_EGL_DEVICE_ID=0`；Ubuntu 基础镜像额外安装 `libegl1`、`libgles2` 后，MuJoCo GPU RGB render 验证通过。
- 上游配置文件在 import 阶段会预先解析与本次 LIBERO 推理无关的 base/ALOHA checkpoint。部署副本将这两处仅用于训练继承的 `load_path` 改为空占位；评测加载器会在模型构建前将其替换为上述本地 LIBERO Policy checkpoint。该兼容性改动不跳过、不修改 Policy 权重。
- LIBERO/robosuite 在进程结束时输出过 EGL context 析构警告；它出现在所有轨迹、成功统计和 MP4 已写完之后。本轮结果保留该已知问题，后续批量实验会显式关闭 renderer 以消除日志噪声。

## 第二阶段：受控鲁棒性评测

只在保持任务、检查点与随机种子不变时改变一个因素：

1. 动作分块长度：`1 / 4 / 8 / 16`；
2. 观测扰动：轻微高斯噪声、亮度变化、随机遮挡；
3. 评测动作 horizon：闭环频率与成功率/推理耗时的关系。

报告均值、成功率、95% binomial Wilson 区间、每回合推理延迟，并保存配置与原始 JSON 结果。

## 实施边界

- RTX PRO 6000 单卡用于官方检查点的推理、闭环仿真与鲁棒性评测。
- 官方全量 LIBERO 训练需要多卡 GPU，不在本项目的单卡复现范围内。
- 权重、数据集、缓存和视频均不纳入 Git；脚本、配置、指标表与复现实验说明纳入版本控制。
