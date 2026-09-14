# PushCube Latent World Model

第一阶段目标：在 ManiSkill `PushCube-v1` 中跑通 GPU 仿真、下载官方示范轨迹，并回放为包含 RGB、机器人状态、动作和奖励的训练数据。

## 推荐云服务器

- Ubuntu 22.04
- NVIDIA RTX 4090 24 GB
- 8 vCPU 或更多
- 32 GB RAM（建议 64 GB）
- 100 GB SSD（建议 200 GB，便于保存 RGB 轨迹和 checkpoint）
- NVIDIA 驱动正常，`nvidia-smi` 可用
- Vulkan 可用，`vulkaninfo --summary` 能识别 NVIDIA GPU

## 1. 创建环境

```bash
conda create -n worldmodel python=3.11 -y
conda activate worldmodel
python -m pip install --upgrade pip
```

先根据云镜像的 CUDA 版本，从 PyTorch 官网选择对应安装命令。随后安装：

```bash
pip install -r requirements.txt
```

当前云端已验证组合：Python 3.10、PyTorch 2.7.1 + CUDA 12.8、ManiSkill 3.0.1、SAPIEN 3.0.3、RTX 5090。PyTorch CUDA wheel 需要按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择匹配的 index；不要让 `pip install torch` 静默替换为 CPU wheel。

## 2. 检查 GPU 仿真

本云实例在无桌面容器中需要使用 EGL Vulkan ICD；不要覆盖系统的 NVIDIA 驱动文件。项目内已提供可回退配置，先在每个新 SSH shell 中执行：

```bash
export XDG_RUNTIME_DIR=/tmp
export VK_ICD_FILENAMES="$PWD/configs/nvidia_icd_egl.json"
vulkaninfo --summary | grep -E 'deviceName|driverName'
```

验收输出必须包含 `NVIDIA GeForce RTX 5090`，不能只有 `llvmpipe`。之后运行：

```bash
python scripts/smoke_test.py --backend physx_cuda --num-envs 64 --steps 120 --video outputs/smoke.mp4
```

成功标准：

- 输出 `PushCube-v1` observation/action 结构；
- 完成 120 步，无 Vulkan 或 CUDA 报错；
- 生成 `outputs/smoke.mp4`；
- 输出平均 step FPS。

## 3. 下载并回放官方示范

```bash
python -m mani_skill.utils.download_demo "PushCube-v1"
```

下载后，根据命令输出定位 `trajectory.h5`，再执行：

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/rigid_body/PushCube-v1/trajectory.h5 \
  --use-first-env-state \
  -b physx_cuda \
  -c pd_joint_delta_pos \
  -o rgb \
  --record-rewards \
  --save-traj \
  --save-video
```

原始示范为压缩轨迹，通常不直接存 RGB；回放步骤会利用初始状态、动作和随机种子重新生成视觉观测。

## 4. 检查数据

```bash
python scripts/inspect_h5.py path/to/replayed_trajectory.h5
```

## 5. 训练第一版 action-conditioned latent dynamics

先用小规模回放数据验证端到端训练是否正确：

```bash
python scripts/train_dynamics.py \
  --data /root/.maniskill/demos/PushCube-v1/motionplanning/trajectory.rgb+state.pd_joint_pos.physx_cuda.h5 \
  --epochs 5 --batch-size 32 --image-size 64 \
  --output outputs/baseline_debug
```

模型同时优化当前帧重建、下一帧预测、latent dynamics 和状态预测。训练完成后会在 output 目录保存 `last.pt` 与 `metrics.json`；指标定义、实验日志和已解决问题见 [`docs/`](docs/)。

下一阶段将在确认字段结构后实现：

1. RGB + proprioception + action 数据加载器；
2. 视觉 encoder 与 action-conditioned latent dynamics；
3. 单步与多步 rollout 训练；
4. 基于 goal cost 的 CEM 动作规划；
5. rollout horizon 和 action-conditioning 消融。

## 简历更新门槛

只有在对应结果真实生成后，才将简历中的“计划”替换为：

- 已跑通 GPU 并行仿真和 RGB 轨迹回放；
- 已形成 observation-action-next observation 数据集；
- 已完成 latent dynamics 训练与多步 rollout；
- 已完成 CEM 闭环控制和成功率评测。
