# PushCube 潜空间世界模型：复现实验命令

以下命令在已验证的 Ubuntu 22.04 + RTX 5090 环境执行。项目、数据和输出均位于数据盘 `/root/autodl-tmp/world_model_sim`。

```bash
cd /root/autodl-tmp/world_model_sim
source /root/autodl-tmp/envs/worldmodel/bin/activate
export XDG_RUNTIME_DIR=/tmp
export VK_ICD_FILENAMES="$PWD/configs/nvidia_icd_egl.json"
```

先确认 Vulkan 使用 NVIDIA 而非软件渲染：

```bash
vulkaninfo --summary | grep -E 'deviceName|driverName'
python scripts/smoke_test.py --backend physx_cuda --num-envs 4 --steps 20
```

## 数据回放

原始官方 `motionplanning/trajectory.h5` 与 JSON 放在 `data/pushcube_1000/`。以下回放会生成 RGB、state、action、reward、success 的 HDF5：

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path data/pushcube_1000/trajectory.h5 \
  --sim-backend physx_cuda --obs-mode rgb+state \
  --use-first-env-state --record-rewards --save-traj --count 1000
```

当前实际输出为 `trajectory.rgb+state.pd_joint_pos.physx_cuda.h5`，回放请求 1,000 条、成功落盘 975 条。检查数据契约：

```bash
python scripts/inspect_h5.py data/pushcube_1000/trajectory.rgb+state.pd_joint_pos.physx_cuda.h5
```

## 视觉世界模型训练与评测

```bash
DATA=data/pushcube_1000/trajectory.rgb+state.pd_joint_pos.physx_cuda.h5

python scripts/train_dynamics.py \
  --data "$DATA" --output outputs/baseline_1000_trajsplit \
  --epochs 20 --batch-size 128 --num-workers 4

python scripts/train_multistep_dynamics.py \
  --data "$DATA" --checkpoint outputs/baseline_1000_trajsplit/last.pt \
  --output outputs/multistep_1000_h5 --epochs 15 --horizon 5 \
  --batch-size 64 --num-workers 4

python scripts/evaluate_rollout.py \
  --data "$DATA" --checkpoint outputs/multistep_1000_h5/last.pt \
  --output outputs/multistep_1000_h5/rollout_metrics.json
```

生成 held-out trajectory 的三栏视频（input / predicted next / ground truth next）：

```bash
python scripts/visualize_predictions.py \
  --data "$DATA" --checkpoint outputs/multistep_1000_h5/last.pt \
  --trajectory traj_291 --frames 64 --fps 10 \
  --output outputs/multistep_1000_h5/traj_291_prediction.mp4
```

## 控制对照

官方示范最短 61 步、中位数 69 步；必须显式传 `--max-steps 100`，默认的 Gym 50-step 截断不能用于此数据集的控制结论。

```bash
python scripts/evaluate_knn.py \
  --data "$DATA" --output outputs/knn_1000_k1_chunk2_eval_20_steps100.json \
  --episodes 20 --neighbors 1 --chunk-size 2 --max-steps 100 --seed 42
```

该命令是特权状态检索的诊断控制基线，不能作为视觉策略或世界模型规划器的成功率。CEM 调用方式如下；当前结果仍存在 model exploitation，仅用于记录和后续改进：

```bash
python scripts/evaluate_cem.py \
  --checkpoint outputs/multistep_1000_h5/last.pt --data "$DATA" \
  --output outputs/multistep_1000_h5/cem_terminal_eval.json \
  --episodes 10 --max-steps 100 --horizon 5 \
  --population 512 --elites 64 --iterations 5 --objective terminal
```

## PickCube flow video world model（proprioception-only）

```bash
PICK_DATA=data/pickcube_1000/PickCube-v1/motionplanning/trajectory.rgb+state.pd_joint_pos.physx_cuda.h5

python scripts/train_flow_video_world_model.py \
  --data "$PICK_DATA" --output outputs/pickcube_flow_proprio_1000 \
  --epochs 30 --batch-size 128 --num-workers 4 --state-dim 25

python scripts/evaluate_flow_video_world_model.py \
  --data "$PICK_DATA" --checkpoint outputs/pickcube_flow_proprio_1000/last.pt \
  --output outputs/pickcube_flow_proprio_1000/rollout_metrics.json \
  --max-starts-per-trajectory 32 --flow-steps 4 --seed 42
```

`--state-dim 25` 仅输入 Panda qpos、qvel 和 TCP proprioception；不要用默认扁平 state 中随后出现的 object / goal pose 作为视觉模型输入。
