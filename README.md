# IsaacLab CRL (Constrained RL) Tasks (Teacher-Student)

面向 Galileo 机器人复杂地形全向移动任务的 FPPO + Teacher-Student 训练与评估说明。

## 安装
```bash
cd /home/lz/Project/IsaacLab/fppo_ts
pip install -e .

cd /home/lz/Project/IsaacLab/fppo_ts/crl_tasks
pip install --no-build-isolation -e .
# 遇到旧版本残留，可先卸载再装：
# pip uninstall -y crl_tasks && pip install --no-build-isolation -e .
```

## 可选：清理缓存
```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
```

## 训练与评估
### 环境准备
```bash
conda activate isaaclab
cd /home/lz/Project/IsaacLab/fppo_ts
```

### 单机单卡
通过 `--algo` 覆盖默认算法（默认使用配置文件内的 `class_name`，当前 Teacher/Student 默认 FPPO）。

可选值：
- `fppo` / `ppo` / `ppo_lagrange` / `cpo` / `pcpo` / `focpo` / `distillation`（使用 `scripts/rsl_rl/algorithms/` 里的实现）

示例：
```bash
# Teacher
LOG_RUN_NAME=fppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo fppo --num_envs 4096 --max_iterations 50000 --run_name teacher --headless \
  --logger wandb --log_project_name galileo_fppo

# Student
LOG_RUN_NAME=fppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Student-v0 \
  --algo fppo --num_envs 4096 --max_iterations 50000 --run_name student --headless \
  --logger wandb --log_project_name galileo_fppo
```

***如果多次用同一个 LOG_RUN_NAME，日志和 checkpoint 会写到同一个目录，可能覆盖或混在一起。***

### 多卡分布式（4 卡示例）
```bash
# 环境
cd /home/lz/Project/IsaacLab/fppo_ts

conda activate isaaclab

# 可选：清理残留
fuser -k -9 /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3

# 安全模式变量（适配 IOMMU/带宽受限机型）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=ens3f3
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --distributed --num_envs 3500 --max_iterations 50000 \
  --run_name galileo-teacher --device cuda:0

```
- `LOG_RUN_NAME` 决定日志目录名：`logs/rsl_rl/<exp>/<LOG_RUN_NAME>_<run_name>`。
- `--num_envs` 为每卡环境数，按显存调整。
- 查看可用环境：`python list_envs.py`。

### 可视化 / 评估 / 部署命令

**Play（可视化回放）**

使用 `--checkpoint` 参数指定要加载的模型文件（`*.pt`），支持相对路径或绝对路径。

```bash
# Galileo 教师模型
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Teacher-Play-v0 \
  --num_envs 50 \
  --checkpoint logs/rsl_rl/galileo_fppo/

# Galileo 学生模型
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Student-Play-v0 \
  --num_envs 50 \
  --checkpoint logs/rsl_rl/galileo_fppo/
  --enable_cameras

```



### git 强制覆盖代码
```bash
git fetch --all
git reset --hard origin/master
```
