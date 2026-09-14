# PickCube Action-Conditioned Flow Video World Model

## 目标

在 ManiSkill `PickCube-v1` 中构建生成式、动作条件的视觉世界模型。与 PushCube 项目的确定性 latent dynamics 不同，本项目以 conditional flow matching 学习从噪声到下一视觉 latent 的连续向量场；推理时通过数值积分采样下一帧 latent，并可自回归展开。

## 数据与协议

- 官方 motion-planning demonstrations：请求回放 1,000 条，RGB + state GPU 回放成功保存 **1,000/1,000**。
- HDF5：`data/pickcube_1000/PickCube-v1/motionplanning/trajectory.rgb+state.pd_joint_pos.physx_cuda.h5`（约 404 MB）。
- 按完整 trajectory 以固定 seed 42 划分：90% train / 10% held-out；不会将相邻视频帧拆分至不同集合。
- 训练图像为 `64 x 64`，128-d latent；CNN autoencoder、state transition head 与 conditional vector field 端到端训练 30 epochs。

> 口径说明：下述首轮 35-d `rgb+state` 结果包含 simulator 提供的 object / goal state，只作为管线调试基线，**不作为简历主结果**。主实验已改为前 25 维 qpos、qvel、TCP 的 proprioception，视觉负责物体状态；其训练与相同评测正在后台运行。

## 35-d simulator-state 调试基线

单步训练末轮 held-out：flow MSE `0.16610`，视觉重建 MSE `0.00349`，state MSE `3.70e-05`。

自由采样 rollout 使用 100 条 held-out trajectories 的 3,200 个起点；每一步以 4 次 Euler 积分采样视觉 latent，并分别输入真实动作或全零动作。

| Horizon | Conditioned pixel MSE | Zero-action pixel MSE | Conditioned state MSE | Zero-action state MSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.003137 | 0.003139 | 0.0000359 | 0.001281 |
| 5 | 0.003199 | 0.003201 | 0.000815 | 0.007804 |
| 10 | 0.003298 | 0.003300 | 0.002457 | 0.016013 |

这个 35-d 调试基线中，10-step 状态误差在动作条件下约为 zero-action 的 15.3%，说明模型确实使用连续 action 预测状态演化。像素 MSE 差距很小，当前解释是固定视角和静态背景占较大比例；后续将加入 object-centric crop、LPIPS 与多样本分布指标，而不是只依赖全图 MSE。

## 25-d proprioception 主实验

主实验只输入扁平 state 的前 25 维：Panda qpos、qvel 和 TCP pose；不输入随后出现的 goal 或 object pose。这样目标与方块信息只能由 RGB 视觉 latent 获取。

训练仍使用相同的 1,000 条轨迹、90/10 trajectory-level split、30 epochs；在 100 条 held-out trajectories、3,200 个起点上测得：

| Horizon | Conditioned pixel MSE | Zero-action pixel MSE | Conditioned proprio state MSE | Zero-action proprio state MSE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.003136 | 0.003137 | 0.0000592 | 0.002867 |
| 5 | 0.003200 | 0.003201 | 0.001216 | 0.016913 |
| 10 | 0.003300 | 0.003301 | 0.003205 | 0.040061 |

在不访问 object / goal 真值的条件下，10-step proprioception state MSE 为 zero-action 的约 8.0%（约 12.5 倍改善）。这是简历和后续对外报告使用的主结果。全图 pixel MSE 仍对动作不敏感，因固定背景占主导；后续会加入 object-centric crop、LPIPS 与多样本分布指标。

## 产物与下一步

- `outputs/pickcube_flow_proprio_1000/last.pt`：主实验 checkpoint；
- `outputs/pickcube_flow_proprio_1000/rollout_metrics.json`：主实验动作消融；
- `outputs/pickcube_flow_proprio_1000/traj_716_flow_prediction.mp4`：主实验 input / flow prediction / ground truth 三栏视频；
- `outputs/pickcube_flow_1000/`：包含 object / goal 的 35-d 调试基线，仅用于排查管线。

下一阶段：增加多样本采样和 best-of-N / distributional metric，改为 object crop / segmentation 加权视频损失，并将 action chunk 与语言或任务目标条件结合。
