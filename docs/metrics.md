# 指标与报告口径

所有数字必须以保存的 `metrics.json`、checkpoint 和评测轨迹为准，不在简历或 README 中预填结果。

| 阶段 | 指标 | 计算方式 | 用途 |
| --- | --- | --- | --- |
| 仿真吞吐 | transitions/s | `num_envs * steps / elapsed_seconds` | 检查 GPU 并行采样 |
| 重建 | RGB MSE | decoder 输出与 `o_t` 的像素均方误差 | 防止视觉 latent 坍缩 |
| 单步预测 | next-image MSE | predicted latent decode 与 `o_{t+1}` 的像素 MSE | 基础视觉预测 |
| 单步动力学 | latent MSE | `f(z_t,s_t,a_t)` 与 encoder 的 `z_{t+1}` | 动作条件转移 |
| 状态预测 | state MSE | predicted `s_{t+1}` 与真实状态 | 动力学一致性 |
| 多步漂移 | rollout MSE | 连续 rollout H 步后与真实观测/状态对比 | 世界模型稳定性 |
| 规划 | success rate | CEM 闭环达到 PushCube 成功条件的回合占比 | 控制有效性 |

必须额外报告两组消融：

1. **action-conditioned vs. observation-only**：后者将动作置零或移除；
2. **rollout horizon**：至少比较 1、5、10 步。
