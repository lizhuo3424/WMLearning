# V-JEPA 2-AC 潜空间能量规划复现

## 目标与边界

复现 Meta FAIR 发布的 V-JEPA 2-AC action-conditioned latent world model 的公开 energy-landscape / CEM-MPC 示例，确认模型能在给定当前视觉状态、末端状态和图像目标时，对候选动作的潜空间预测能量作出有方向性的区分。

这不是在仿真器或真实机器人上的任务成功率评测，也不表述为自行训练了 V-JEPA 2-AC。这里报告的是官方预训练模型在一条公开 Franka 轨迹上的接口复现与时序诊断。

## 可复现环境

| 项目 | 实际配置 |
| --- | --- |
| 官方源码 | `facebookresearch/vjepa2` main archive，commit `204698b45b3712590f06245fbfba32d3be539812` |
| Python / Runtime | Python 3.11.16；PyTorch `2.7.0+cu128` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition（约 96 GB VRAM） |
| Checkpoint | `vjepa2-ac-vitg.pt`，`11,760,743,310` bytes |
| Checkpoint SHA-256 | `0b5e3c4bf77a473cd8c61d32fbd87b28cdbba043fb3b8267f3b8bcfb1d5b9e6b` |
| 模型装载验证 | encoder + AC predictor 合计 `1,317,394,944` 参数；装载后约 `5.12 GiB` GPU memory |

当前官方主分支的 `src/hub/backbones.py` 将默认权重 URL 指向 `localhost:8300`，而公开 Meta URL 被注释。部署副本将该 URL 恢复为 `https://dl.fbaipublicfiles.com/vjepa2`，并保留 `backbones.py.before_public_url_patch` 备份；模型实际加载使用本地、已完成 SHA-256 校验的 checkpoint。

## 轨迹与输入

官方 notebook 引用了但未随 GitHub 源码提交的 `franka_example_traj.npz`。实验从公开镜像 `ckadirt/vjxla` 获取该同名样例，并验证其文件 SHA-256 为 `99e31cbd4734cbcaeeeb921af8caed7e7ce4a9b28b005b05cac589b88fe34911`，与镜像 LFS 指针一致。

样例输入为：

- RGB observation：`(1, 2, 256, 256, 3)`；
- 机器人 pose / gripper state：`(1, 2, 7)`；
- 视觉预处理、latent normalization、7-DoF action/state interface 均沿用官方 notebook；
- 候选动作固定旋转与 gripper 分量为 0，只搜索 Cartesian `x/y/z` 位移。

## 实验 A：正向 energy landscape

对 `[-0.075, 0.075]^3` 的 Cartesian 网格执行 `5 x 5 x 5 = 125` 个单步动作候选。每个候选由 AC predictor 预测下一帧 latent token，并以其和轨迹目标帧 target token 的平均 L1 距离作为 energy。

| 指标 | 结果 |
| --- | --- |
| Energy 最小 / 最大值 | `0.412481 / 0.487973` |
| 最小 energy 候选 `xyz` | `[+0.075, +0.075, +0.075]` |
| 记录的真实下一步 `xyz` | `[+0.09237, +0.03099, +0.08428]` |

三轴方向一致；`x/z` 到达搜索网格上界，说明这个低分辨率网格不足以精确恢复幅度。原始结果：云端 `outputs/official_energy_grid_5x5x5.npz`。

## 实验 B：时间反转诊断

按照官方 notebook 的 `play_in_reverse` 说明，反转同一段 observation/state 序列，在完全相同的 125 候选网格上重新计算能量。

| 指标 | 结果 |
| --- | --- |
| Energy 最小 / 最大值 | `0.432031 / 0.492142` |
| 最小 energy 候选 `xyz` | `[-0.075, -0.075, -0.075]` |
| 反转后的真实下一步 `xyz` | `[-0.09237, -0.03099, -0.08428]` |

最优候选随轨迹方向从全正变为全负，且三轴符号均与反转后的真实动作一致。这是 action-conditioned temporal directionality 的单轨迹诊断，不是跨任务泛化结论。原始结果：云端 `outputs/official_energy_grid_reversed_5x5x5.npz`。

## 实验 C：低预算 CEM-MPC

使用官方 notebook 的低预算设置：rollout `2`、samples `25`、top-k `10`、CEM steps `2`、Cartesian max norm `0.075`。为利用本地 GPU，执行设备从 notebook 的 CPU 改为 CUDA；优化器设置不变。

| 项目 | `xyz` |
| --- | --- |
| CEM 第一步规划 | `[+0.03835, +0.01502, +0.05571]` |
| 记录的真实下一步 | `[+0.09237, +0.03099, +0.08428]` |

规划第一步与记录动作三轴方向一致，但幅度更保守。由于该 CEM 配置将旋转分量固定为 0，比较只针对 `xyz`；不能将其解释为完整 7-DoF action imitation 精度。原始结果：云端 `outputs/official_cem_plan_lowbudget.npz`。

## ManiSkill 跨仿真器准备

为避免将不同控制接口混为一谈，跨仿真器部分单独使用 ManiSkill `3.0.1` 的 `PickCube-v1`：

- 已验证 headless CPU physics + GPU Vulkan 渲染、RGB 观测与一步仿真：`base_camera` RGB 为 `(1, 128, 128, 3)`，`rgb_array` 为 `(1, 512, 512, 3)`；
- SAPIEN 未发现系统级 `libvulkan` / NVIDIA ICD 时回退到其内置 Vulkan，实际 reset、step 与渲染均成功；该警告须保留在环境记录中，不能写作“无警告部署”；
- 在 `sensor_configs={width: 256, height: 256}` 下，观测已验证为 `(1, 256, 256, 3)`；
- 控制器明确选用 `pd_ee_delta_pose`，其动作空间为 `7` 维（位移 3、姿态增量 3、夹爪 1），而非默认 `pd_ee_delta_pos` 的 4 维控制；TCP position 和 quaternion 分别可从环境读取为 `(1, 3)` 和 `(1, 4)`。

## 实验 D：ManiSkill PickCube 跨域 action-energy pilot

下载官方 `haosulab/ManiSkill_Demonstrations` 数据集的 `PickCube-v1` 演示包（经公开镜像获取）。ZIP 大小 `36,590,010` bytes，SHA-256 验证为 `b2d4afb30fa309755862b98c342e6ee18918253c93f3bbac16ed6670748f26d8`。其中 `pd_ee_delta_pose` 轨迹有 1,013 个 episode、每条最多 50 步、原始 action shape 为 `(50, 7)`。

原始 action 是控制器命令，数值可超出归一化 action-space，故不直接送入 V-JEPA。每个 transition 用记录 state 恢复 CPU physics，以 SAPIEN Vulkan 渲染 `256 x 256` RGB；再从 TCP 真实位置、Euler `xyz`、归一化夹爪闭合度构造 7D pose，并以与官方 notebook 相同的相邻 pose 差分构造 action。

在 5 个不同 episode 上，将专家 action、全分量反向 action、zero action 分别送入同一个 V-JEPA 2-AC predictor；能量为预测最后一帧 latent token 与下一 RGB 帧 goal token 的平均 L1 距离。

| Episode / step | Expert | Inverse | Zero | 最低 energy |
| --- | ---: | ---: | ---: | --- |
| `0 / 0` | `0.379520` | `0.366624` | `0.354861` | Zero |
| `10 / 10` | `0.320468` | `0.326215` | `0.321858` | Expert |
| `100 / 20` | `0.310402` | `0.310308` | `0.310276` | Zero |
| `500 / 30` | `0.300842` | `0.301034` | `0.300924` | Expert |
| `1000 / 40` | `0.311598` | `0.311384` | `0.311486` | Inverse |

专家 action top-1 为 `2 / 5 = 40%`。尺度扫描 `{-1,-0.5,0,0.25,0.5,1,1.5}` 中，5 个样本的最低点依次为 `0`、`1`、`-0.5`、`1.5`、`-1`；只有一个样本偏好原始专家幅度。因此不能以单一 action 缩放解释失配。将同一组 transition 时间反向并输入反向 pose delta 后，directed-action top-1 仍为 `2 / 5 = 40%`，没有重现官方 Franka 样例中的稳定方向翻转。

为排除微小动作被 zero baseline 主导，用旋转矩阵相对变换（而非直接相减 Euler angle）在前 30 个 episode 中筛选出 5 个约 `5--6.5 cm` 平移的大动作 transition 后重新评测；5 个样本均由 zero action 取得最低 energy，expert top-1 为 `0 / 5`。这说明早先的不稳定排序不是简单的小动作问题。固定相机、场景外观、机器人形态和状态坐标系与 V-JEPA 2-AC 的 DROID/Franka 域不同：接口可端到端执行，但未经适配的零样本动作排序不可靠。这不是 PickCube 任务成功率。原始输出：云端 `outputs/maniskill_pickcube_cross_domain_energy_pilot.npz`、`outputs/maniskill_pickcube_action_scale_sweep.npz`、`outputs/maniskill_pickcube_bidirectional_energy.npz`、`outputs/maniskill_pickcube_motion_stratification_rotmat.npz` 与 `outputs/maniskill_pickcube_large_motion_energy.npz`。

该实例可完成 CPU physics + GPU Vulkan RGB 状态回放；SAPIEN 使用内置 Vulkan 并发出系统 ICD 缺失警告。使用 ManiSkill `cuda` / `physx_cuda` physics 后端时，首次场景初始化长时间无 GPU 利用率且未返回，已终止该单个卡住进程；因此本文不声称使用 GPU physics。

## 实验 E：冻结 backbone 的 PickCube 动作残差适配 pilot

为检验上述跨域失配是否能由轻量动作校正缓解，先将 160 条成功 state-replay episode 渲染为 `256 x 256` RGB，共 8,160 帧；按 episode 划分，前 128 条作为训练集，最后 32 条作为严格留出集。另有 10 条官方演示因 actor state 长度不足 51 帧而被排除，不将其混入训练或验证。

V-JEPA 2-AC encoder 和 predictor 全程冻结，仅训练一个 `17 -> 96 -> 7` 的 SiLU MLP（共 `2,407` 参数）。它以记录的 7D action、TCP `xyz`、Euler angle 的 sin/cos 和夹爪闭合度为输入，输出有界残差；优化目标为预测下一帧 latent token 与编码目标 token 的 L1 加一个很小的残差正则。该实验是 action-interface adaptation，不是 V-JEPA 微调，也不构成任务策略训练。

训练 200 step 后，以随机种子固定的 256 个留出 transition 对原 action 与适配后 action 做逐样本比较：

| 指标 | 原 action | 适配后 action |
| --- | ---: | ---: |
| 平均 latent L1 | `0.327068` | `0.326045` |
| 变化 | --- | `-0.001023`（约 `-0.31%`） |
| 单样本 L1 下降比例 | --- | `85.94%` |
| 平均绝对动作残差 | --- | `0.01510` |

同一固定 transition 上进一步将 expert、inverse、zero 同时作为候选动作，得到：

| 排序指标 | 原 action | 适配后 action |
| --- | ---: | ---: |
| Expert 为三者最低 energy（top-1） | `22.66%` | `30.47%` |
| Expert energy 低于 zero | `26.95%` | `39.84%` |
| Expert energy 低于 inverse | `49.61%` | `62.89%` |

这表明适配器在同一渲染/控制接口内学到微弱而一致的校正，并在随机留出 transition 的三候选排序上带来可测提升；但不能据此宣称解决了跨域控制。在实验 D 的同一组 5 个 `5--6.5 cm` 大动作 transition 上，重新对 expert、inverse、zero 三个候选动作都施加适配器后，最低 energy 仍全部为 zero，expert top-1 仍为 `0/5`。因此该 pilot 的结论是：轻量 action residual 对 held-out latent prediction 和常规 transition 排序有小幅收益，但不足以修复 V-JEPA 2-AC 在 ManiSkill 相机、外观及状态坐标域上的困难动作排序失配。

原始输出：云端 `checkpoints/pickcube_action_adapter_pilot.pt`、`outputs/pickcube_action_adapter_pilot_fixed_eval.npz`、`outputs/pickcube_action_adapter_pilot_report.json`、`outputs/pickcube_action_adapter_pilot_fixed_ranking.npz`；训练/评测脚本为 `scripts/train_pickcube_action_adapter_pilot.py`、`scripts/eval_pickcube_action_adapter_pilot.py` 与 `scripts/eval_pickcube_action_adapter_ranking.py`。

## 实验 F：pairwise ranking action-interface calibration

实验 E 的 latent L1 目标并不直接要求 expert action 的能量低于反向或 zero action。因此保持 V-JEPA 2-AC 冻结、复用同一个 `2,407` 参数 adapter，在 128 个训练 episode 上加入候选动作排序约束：

\[
\mathcal{L}=E(a_{\mathrm{expert}})+0.5\,[\max(0,m+E_e-E_i)+\max(0,m+E_e-E_0)]+0.005\Vert\Delta a\Vert_2^2,
\]

其中 margin `m=0.003`，`E_i` 和 `E_0` 分别对应 inverse 与 zero candidate 的 latent goal energy。共训练 600 step，并按另一个固定随机种子在相同 32 条留出 episode 中抽取 128 个 transition 作 checkpoint 选择；最佳 checkpoint 出现在 step 550，监控 top-1 为 `65.63%`。

随后在先前建立的 256 个固定审计 transition 上比较三种 action interface：原始 action、实验 E 的 L1-only adapter、以及本实验的 ranking adapter。

| 指标 | 原 action | L1-only adapter | Ranking adapter |
| --- | ---: | ---: | ---: |
| Expert 为三者最低 energy（top-1） | `22.66%` | `30.47%` | `51.56%` |
| Expert energy 低于 zero | `26.95%` | `39.84%` | `57.42%` |
| Expert energy 低于 inverse | `49.61%` | `62.89%` | `75.00%` |
| Expert latent L1 | `0.327068` | `0.326045` | `0.325937` |

排序目标在常规留出 transition 上有明显收益，但证据边界必须说清：checkpoint monitor 和 256 transition 审计集都来自同一 32 个留出 episode；两个固定样本集合有 `18` 个 exact transition 重叠（审计集有 237 个 unique transition，monitor 有 119 个 unique transition）。因此该表是同域、同 episode pool 内的二次评测，而不是完全独立的泛化测试。

更关键的是，在实验 D 的同一组 5 个大动作 hard case 上，ranking adapter 后仍是 zero action 全部最低 energy，expert top-1 仍为 `0/5`。这说明 ranking calibration 可以改善常规样本的候选动作选择，却没有消除相机、视觉外观、运动尺度与状态坐标域变化造成的困难样本失配；不能写成任务控制成功。原始输出：云端 `checkpoints/pickcube_action_rank_adapter.pt`、`outputs/pickcube_action_rank_adapter_fixed_eval.npz`、`outputs/pickcube_action_rank_adapter_fixed_ranking.npz`；脚本为 `scripts/train_pickcube_action_rank_adapter.py`、`scripts/eval_pickcube_action_rank_adapter.py` 与 `scripts/eval_pickcube_action_rank_adapter_ranking.py`。

为消除该 episode-pool 重叠，进一步固定重划分为：episode `0--127` 训练、`128--143` 只用于 checkpoint 选择、`144--159` 只用于 final audit。使用相同训练配方，最佳 checkpoint 仍为 step 550。在 audit episode 上的 256 个固定 transition 中，expert top-1 从 `24.22%` 升至 `50.00%`，expert 低于 zero 从 `27.73%` 升至 `55.47%`，expert latent L1 从 `0.324865` 降至 `0.324002`；这提供了 episode-level 独立的同域 action-ranking 证据。5 个跨域大动作 hard case 仍为 `0/5`，故结论边界不变。原始输出：云端 `checkpoints/pickcube_action_rank_adapter_split.pt`、`outputs/pickcube_action_rank_adapter_split_fixed_ranking.npz`、`outputs/pickcube_action_rank_adapter_split_fixed_eval.npz`。

## 下一步

1. 将显著位移、闭合夹爪等困难 transition 单独报告，避免随机样本掩盖 zero baseline；
2. 加入 RGB time-reversal 与 goal-frame shuffle 对照，区分视觉时序失配和 action/pose 失配；
3. 保留失败视频和 per-axis 误差，不将官方 Franka 单轨迹结果外推为仿真任务成功率。
