#!/bin/bash
# 激活环境
# srun -N 1 -w byta6000i0,byt4090i0,byt4090i1 --gres=gpu:1 --pty bash
# conda activate Dexre
# 执行传进来的 Python 脚本

python3 detect_from_video.py \
  --robot-name soft \
  --video-path data/hand_video_2.mp4 \
  --retargeting-type dexpilot \
  --hand-type right \
  --output-path data/soft_joints.pkl \
  --debug-joint-vis-dir data/debug/soft_hand_joint_vis \
  --debug-joint-vis-every-n 1 \
  --debug-joint-vis-max-frames 549

python3 render_robot_hand.py \
  --pickle-path data/soft_joints.pkl \
  --output-video-path data/soft.mp4 \
  --headless \
  --apply-soft-hand-panda-mount
