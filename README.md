# WMLearning：具身世界模型实验

面向世界模型与具身智能的实验仓库，包含 ManiSkill 动作条件动力学与 Flow Matching 模型训练、Cosmos Policy 的 LIBERO 闭环评测，以及 V-JEPA 2-AC 潜空间动作评估与轻量适配。记录数据划分、实验配置、实际指标和失败案例。

## 已完成实验

### 1. PushCube 潜空间世界模型与控制对照

- GPU 回放官方 motion-planning 示范，成功保存 975 条 RGB、state、action 轨迹；
- CNN autoencoder + action-conditioned latent dynamics，完成单步训练和 5-step rollout 微调；
- 在 98 条 held-out trajectory、5,888 个起点上，10-step state MSE：action-conditioned `0.005832`，zero-action `5.358248`；
- 修复官方示范时长与环境默认 50-step 截断不一致的问题，统一按 100-step 控制协议评测；
- 特权状态 kNN action-chunk 检索基线（chunk=2）在固定 20 seeds 达到 70% success rate、平均最终 cube-goal 距离 `0.120881 m`。

### 2. PickCube Action-Conditioned Flow Video World Model

- GPU 回放 1,000/1,000 条官方 motion-planning 示范；
- CNN autoencoder + conditional flow matching：通过当前视觉 latent、25-d proprioception（qpos/qvel/TCP）和连续动作条件化向量场，从噪声积分采样下一视觉 latent；
- 不使用 simulator object / goal state。在 100 条 held-out trajectory、3,200 个起点上，10-step proprioception state MSE：action-conditioned `0.003205`，zero-action `0.040061`；
- 生成 input / prediction / ground-truth 三栏自回归 rollout 视频。

### 3. Cosmos Policy + LIBERO 闭环评测

- 在 NVIDIA 公开的 `Cosmos-Policy-LIBERO-Predict2-2B` 上完成四套标准 LIBERO 的单种子闭环评测：Spatial / Object / Goal / LIBERO-10 分别为 `50/50`、`50/50`、`50/50`、`47/50`，合计 `197/200`；每套均为 10 个任务各 5 条真实闭环轨迹。该结果不等同于官方跨 seed、每任务 50 条的完整复现；
- 完成 RTX PRO 6000 的 CUDA、FlashAttention、EGL/MuJoCo 无头渲染链路验证，并保存每条真实 rollout 与 future-image 输出视频；
- 在相同 checkpoint、任务、初始状态与随机种子下，执行 horizon `16 / 12 / 8` 的成功率为 `100% / 36% / 2%`（各 50 条）；horizon `4` 的成本受限 pilot 为 `0/14`，揭示策略对训练时动作执行时序的强敏感性；
- 在 horizon `16` 下，双视角同时变暗至 `0.6` 为 `50/50`；双视角中心各遮挡约 `9%` 面积为 `47/50`，提高至约 `25%` 面积则降至 `8/50`，得到清晰的视觉遮挡失效边界。

### 4. V-JEPA 2-AC 潜空间能量规划

- 复现 Meta FAIR V-JEPA 2-AC 官方 action-conditioned world model 的 Franka 轨迹 energy landscape 与低预算 CEM-MPC 接口；加载已校验的 `1.317B` 参数模型，不将官方预训练权重误写为自训练；
- 在 `5×5×5` Cartesian action grid 上，正向轨迹的最优候选与记录动作 `xyz` 方向一致；时间反转后最优候选的三轴符号同步翻转，形成 action-conditioned temporal directionality 的受控诊断；
- 接入 ManiSkill 3.0.1 PickCube 的官方 `pd_ee_delta_pose` 演示，完成 state-replay RGB、TCP 7D pose/action 映射和 expert / inverse / zero action 能量对照；随机 pilot 的 expert top-1 为 `2/5`，旋转矩阵分层筛出的 5 个大位移样本为 `0/5`，量化了未适配相机与机器人域下的不可靠零样本排序；
- 在 128 条状态回放轨迹上训练并冻结评测一个仅 `2,407` 参数的动作残差适配器：加入 pairwise action-ranking loss 后，在从未参与训练或 checkpoint 选择的 16 条 audit episode 上，expert top-1 从 `24.2%` 提升至 `50.0%`，expert 优于 zero 从 `27.7%` 提升至 `55.5%`；但大动作 hard case 仍为 `0/5`，且结果不是 PickCube 任务成功率；
- 该结果是公开轨迹上的方法复现与诊断，非仿真任务成功率；完整配置、权重哈希、负结果与环境限制见实验记录。

## 重要口径

- 所有训练/验证切分均按完整 trajectory 进行，避免相邻帧泄漏；
- `rgb+state` 的扁平 state 含 simulator object / goal pose。PickCube 主实验严格只取前 25 维 proprioception；
- kNN 是特权状态诊断基线，不应表述为视觉策略或世界模型规划成功率；
- 当前 CEM 存在 model exploitation，结果与失败边界都保留在实验日志中，不将其包装为有效控制器。

## 快速开始

建议 Ubuntu 22.04、NVIDIA GPU 和可用 Vulkan。先安装匹配 CUDA 的 PyTorch，再安装依赖：

```bash
pip install -r requirements.txt
export XDG_RUNTIME_DIR=/tmp
export VK_ICD_FILENAMES="$PWD/configs/nvidia_icd_egl.json"
python scripts/smoke_test.py --backend physx_cuda --num-envs 4 --steps 20
```

完整的数据回放、训练和评测命令见 [docs/reproduce.md](docs/reproduce.md)。

## 文档

- [PushCube 实验日志与失败分析](docs/experiment_log.md)
- [PickCube flow world model 实验记录](docs/pickcube_flow_experiment.md)
- [Cosmos Policy + LIBERO 闭环评测记录](docs/cosmos_policy_libero_plan.md)
- [V-JEPA 2-AC 潜空间能量规划复现](docs/vjepa2_ac_experiment.md)
- [复现实验命令](docs/reproduce.md)

大型 HDF5 数据、checkpoint 和 MP4 视频被 `.gitignore` 排除；它们需在本地或云端按文档生成。
