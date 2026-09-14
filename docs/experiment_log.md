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
