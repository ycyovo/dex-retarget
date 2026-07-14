# Dexterous Hand Retargeting & Simulation - AI Coding Guide

## 1. 项目背景 (Project Context)
本项目基于 `dex-retargeting` 框架开发。
**核心目标**：从视频中提取人手运动轨迹，并将其重定向到我们自定义的灵巧手模型上，最终在仿真里重现与视频一致、语义合理的手部姿态。
**当前阶段目标**：在已完成 `soft_hand` 单手 retargeting 初步验证的基础上，继续推进实时视频流到仿真中灵巧手运动的闭环验证。

## 2. 核心任务里程碑 (Milestones)
请在执行任务时，严格按照以下里程碑一步步推进，未完成当前里程碑前，不要跳跃到下一步：

- [x] **Task 1: 自定义灵巧手集成**
  - 将我们的自定义灵巧手模型引入 `dex-retargeting` 框架的手部列表中。
  - 资产路径：`assets/robots/hands/soft_hand` (包含 URDF 等文件)。
- [x] **Task 2: 灵巧手重定向测试**
  - 已完成 `soft_hand` 在 `dex-retargeting` 中的初步 retargeting 验证。
  - 当前视频结果整体已经 make sense：手掌不再全程异常闭合，四根长指的横向顺序和屈伸语义基本正确。
  - 最终关键修复是在 `detect_from_video.py` 中为 `soft_hand + DEXPILOT` 增加固定坐标系旋转：`MediaPipe wrist frame -> soft_hand/base_link frame`。
  - 该旋转使用 semantic `palm_frame` 估计：四根长指平均方向作为 forward，`index_tip - little_tip` 作为 lateral，避免 Kabsch 类点云拟合导致四指横向镜像。
  - 新增验证目标：基于旋转后的 21 个 MediaPipe 手部关键点坐标，生成类似手部骨架图的可视化（标注点编号并按手指连线），用于直观检查坐标旋转后的手掌朝向、四指顺序和拇指位置是否正确。
  - **Bonus（非必要）**：当前 `soft_hand` retargeting 结果中，个别手指仍可能出现交叉、错位或局部顺序不自然的问题。后续可尝试从 fingertip 目标点、手指间距/碰撞约束、关节限位、DexPilot pairwise 向量权重等方向小步排查并改善，但不阻塞 Task 3/4。
  - 一键回归：`tests/test_soft_hand_task2_pipeline.py`（先 `source`/执行项目约定的 `run_test.sh` 再跑 Python）。
  - 主要相关文件：
    - 数据生成：`example/vector_retargeting/detect_from_video.py`
    - 渲染写入：`example/vector_retargeting/render_robot_hand.py`
    - 重定向配置：`src/dex_retargeting/configs/teleop/soft_hand_right_dexpilot.yml`
    - 渲染用 URDF：`assets/robots/hands/soft_hand/soft_hand_right_hand_only.urdf`
- [ ] **Task 3: 实时视频流 Retargeting**
  - 将离线链路改造成实时链路：`视频流/摄像头 -> MediaPipe 手部检测 -> soft_hand DEXPILOT retargeting`。
  - 复用 Task 2 中已验证的 `MediaPipe wrist frame -> soft_hand/base_link frame` 坐标系旋转逻辑，避免离线和实时结果语义不一致。
  - 先实现最小可用版本：单线程或简单同步循环即可，优先保证每帧输出 qpos 正确、异常帧可跳过、日志清晰。
- [ ] **Task 4: 实时仿真可视化验证**
  - 跑通完整实时链路：`实时视频流 -> 提取手部信息 -> 运动重定向 -> SAPIEN/仿真中 soft_hand 实时运动`。
  - 重点验证实时性、稳定性和视觉语义：手指屈伸方向、四指顺序、延迟、抖动、丢帧恢复，以及手指交叉/错位是否影响可用性。
  - 优先输出可视化结果和必要日志；如需保存结果，再增加可选录制视频或 qpos 轨迹功能。

## 3. 技术与编码限制 (AI Coding Constraints & Rules)

为了保证项目的稳定性和代码的可靠性，AI 在生成代码时必须遵守以下限制：

### 3.1 机器人与仿真领域限制 (Robotics Specific)
1. **URDF 处理**：在修改 URDF 文件时，务必注意 `joint` 的类型、`limits`（关节限位）以及 `parent/child link` 的层级关系；当前阶段不再要求机械臂与灵巧手拼接。
2. **坐标系与变换**：在进行运动学重定向时，任何涉及坐标系转换（Transformations）、四元数（Quaternions）或欧拉角（Euler angles）的代码，**必须在代码注释中明确写出当前是从哪个坐标系转换到哪个坐标系**。
3. **碰撞与自碰撞**：在仿真测试中，必须考虑到物理穿模问题。在写测试脚本时，需开启或保留检查物理碰撞（Collision）的逻辑。

### 3.2 Vibe Coding 规范 (Workflow Specific)
1. **Baby Steps（小步快跑）**：绝对不要一次性重写整个核心文件。每次只修改一个函数或一个小模块，修改后必须提供一段极简的测试代码验证。
2. **防御性编程**：由于 `dex-retargeting` 涉及复杂的优化算法（如非线性优化），经常会出现无解或奇异点。AI 在写代码时必须加入 Try-Catch，并打印清晰的 Warning 日志，而不是让程序直接 Crash。
3. **日志与可视化优先**：在编写 Task 2 和 Task 4 的测试代码时，优先使用现有的可视化工具（如 OpenCV 窗口, PyBullet/SAPIEN 的 GUI 等）输出视觉结果。对于我们来说，“看着对不对”比数据更直观。
4. **不要虚构方法（No Hallucinations）**：如果你不确定 `dex-retargeting` 框架中是否包含某个特定的类或方法，请先让我（人类）运行脚本打印该模块的 `dir()`，或者让你读取框架的源码文件，**禁止猜测不存在的 API**。
5. **导入阶段卡住就停（Import Blockers）**：Vibe Coding 时若测试卡在**导入阶段**（如 `ModuleNotFoundError`、conda 环境未激活、SAPIEN/GPU/Vulkan 初始化失败、沙箱无法加载依赖等），**不要长时间反复试跑或纠结环境**。说明卡点与已做排查后，**停止自动测试**，交给我（人类）在正确机器/环境下执行；继续推进代码改动或给出待我本地验证的命令即可。

## 4. 目录结构约束
- 请保护 `SoftHandGrasping/task/assets/` 目录下的原始网格（meshes）文件，不要尝试修改 `.obj` 或 `.stl` 等 3D 文件本身，只能修改或生成对应的 URDF/XML 描述文件。
- 所有新生成的测试脚本，请统一存放在 `tests/` 目录下（如不存在则创建）。
