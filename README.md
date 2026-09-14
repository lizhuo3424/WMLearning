# WMLearning：具身世界模型实验

面向世界模型 / 具身智能算法实习的可复现实验仓库。项目以 ManiSkill 3 的官方机器人操作示范为数据源，覆盖确定性潜空间动力学与生成式 action-conditioned flow video world model 两条路线。

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
- [复现实验命令](docs/reproduce.md)

大型 HDF5 数据、checkpoint 和 MP4 视频被 `.gitignore` 排除；它们需在本地或云端按文档生成。
