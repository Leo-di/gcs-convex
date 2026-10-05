# 24：碰撞反例驱动的区域收紧与预抓取路径修复

本轮聚焦上一轮剩余两例，而非重新执行全部基准。

## 碰墙：区域收紧后重新求解

杯子左移 x115_y5 的携物轨迹碰墙。将 IRIS 粒子数从 220 提高到 880、迭代上限从 3 提高到 6 后，仍出现约 2.76 mm 碰撞。该负结果保留为 left_stronger。

新增有界反例循环：对 GCS 输出路径按每段 10 等分检查，找到最深穿透状态，在原附着杯子场景计算距离对 10D 状态的有限差分梯度；只在能够保留区域原始种子的区域上增加分离半空间，再求解 GCS。最多 8 轮，切不动或求解失败时拒绝。局部线性化不能证明整个区域无碰撞，最终仍以独立完整轨迹审计为准。被修改区域重新抽样审计。

本例一轮收紧后通过完整审计，加密轨迹最小环境间隙约 2.18 mm；区域抽样为 0/3000 碰撞。零抽样碰撞不是连续安全证明。

## 碰杯：实际发生于移动到预抓取姿态

补存首次违反帧后发现，杯子右移 x115_y5 在 arm_to_pregrasp 阶段碰杯。仅将后一段 approach_cup 改为笛卡尔直线没有解决问题，此负结果保留。

新增固定底盘的 7D 关节空间避障搜索，使用原关节限位、环境几何以及不受原碰撞过滤影响的杯子—夹爪距离检查；随机种子 1702，搜索最多 3000 次或 20 秒。首版只检查不穿透，仍在最终加密插值审计发现约 1.38 mm 穿透，失败报告保留为 no_margin_failure.json。增加 2 mm 指部间隙后，完整轨迹审计通过，最小环境间隙约 2.47 mm。该预抓取选项目前需通过 collision_checked_pregrasp 显式开启，并未推广为所有动作的通用规划器。

本例区域抽样仍有 1/3200 碰撞；控制案例也有 1/2800。反例循环修复的是当前优化路径暴露的碰撞，不保证消除区域中所有潜在碰撞。

## 回归与结果边界

原有完整流程测试发现 nominal x120_y-10 也会在新运输管线中出现轨迹碰撞。加入默认反例检查后，该测试恢复通过，相关测试共 14/14 通过；对应控制案例一轮收紧后通过加密审计。没有修改测试验收阈值。

两条目标轨迹的动力学结果另见同目录 dynamics/result.json。实验包含：left_stronger、right_original、right_cartesian、control_cartesian、left_cuts、right_pregrasp、control_cuts。只在接受的独立审计结果上执行物理回放；失败记录均保留。

未重新跑全部 36 个提案或 B1/B2/joint 时间基准，因此不汇报新的总体成功率或性能优势。之前 6 条成功与本轮 2 条恢复来自不同修订版本，不能声称同一最终版本已经完成八条全量验证。

代码：src/refine_carry_regions.py、src/safe_pregrasp.py、src/refine_failed_transport.py。数据：outputs/results/remaining_collision_refinement/。执行仍为理想驱动底盘、自由杯子接触仿真，不代表硬件或连续时间安全验证。

实际接触执行结果：{"left_cuts": {"satisfied": true, "unexpected_contact_count": 0, "cup_attachment": false, "cup_pose_resets_after_initialization": 0}, "right_pregrasp": {"satisfied": true, "unexpected_contact_count": 0, "cup_attachment": false, "cup_pose_resets_after_initialization": 0}}。两条均按相同接触验收标准记录。
