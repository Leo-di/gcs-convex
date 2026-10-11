# 无全臂语义约束的 GCS 反例复现

日期：2026-10-11

## 目的

复现一个具体反例：普通的几何碰撞约束允许 GCS 找到连续、无碰撞的轨迹，但轨迹在任务模式切换前已经让机械臂越过房间语义边界，因而不满足“操作阶段整机留在房间内”的直觉要求。这里的“无语义约束”准确指**移除了全臂房间约束**；底盘的 room/transit 模式、场景碰撞约束、起终点仍然保留。

## 固定问题

- 原始 Drake 双房间场景；机械臂保持张开夹爪，不携带杯子。
- 规划状态为完整记法 `q=(x,y,yaw,q₁,…,q₇)∈R¹⁰`，但实验只在固定 `y=0`、`yaw=0` 和一个耦合姿态族 `q_arm=q_home+α(q_fold−q_home)` 上优化。因此区域内在维数为 2（底盘 `x` 与收臂参数 `α`），不是通用的 10D 全空间规划。
- 起点 `x∈{−1.3,−1.1}`，终点 `x=0.6`；模式切换点 `g∈{−0.6,−0.5}`。在 `room` 模式底盘必须满足 `x≤g`；在 `transit` 模式允许向另一侧通行。
- 全臂语义边界是平面 `x=0`。语义谓词要求 room 模式期间，机械臂与夹爪全部碰撞几何都位于 `x≤0`。碰撞几何检查在两种方法中都启用。

## 对照方法

1. **Base-only GCS**：保留底盘模式、固定起终点和碰撞约束，移除 room 模式下的全臂边界约束。
2. **Postcheck**：复用 Base-only 的同一条路径，只做全臂语义事后检查；违规就拒绝，不重规划。
3. **Semantic GCS**：将全臂语义边界转成保守的凸区域收紧条件，再由 Drake `GraphOfConvexSets` 和 Clarabel 规划。源码用运动学 Lipschitz 上界约束姿态区间内的机械臂支撑范围。

三种方法共用同一碰撞筛选区域、起终点、状态族、成本及求解器；变化项是全臂语义是否参与 GCS 区域和连通图。

## 新一轮运行结果

本次独立运行了 5 个条件 × 3 种方法，共 15 次规划。前四项为正例，限制姿态项为预设负例。

| 起点 `x` | 模式切换 `g` | Base-only 碰撞最小间距 | Base-only 全臂语义最小 residual | Semantic GCS residual | 规划审核 |
|---:|---:|---:|---:|---:|---|
| −1.3 | −0.6 | 1.505 cm | −19.06 cm | +2.83 cm | baseline 违规；semantic 通过 |
| −1.3 | −0.5 | 1.505 cm | −29.06 cm | +9.74 cm | baseline 违规；semantic 通过 |
| −1.1 | −0.6 | 1.505 cm | −19.06 cm | +2.83 cm | baseline 违规；semantic 通过 |
| −1.1 | −0.5 | 1.505 cm | −29.06 cm | +8.34 cm | baseline 违规；semantic 通过 |

前四项里 Base-only 均求得连续轨迹，密集离散审核均未发现碰撞，但全臂语义 residual 为负；这正是“几何可行、语义不合意”的反例。Postcheck 对这四条相同路径全部拒绝，说明仅在求解后查规则不能自动修复轨迹。Semantic GCS 四项均求解并通过全臂语义和碰撞审核。该轨迹审核每条使用 343–418 个插值状态；最小碰撞距离约 1.505 cm。

受限姿态负例规定 `α≤0.15`。Semantic GCS 在该离散凸覆盖图中不可达；这仅表示此姿态族和当前区域覆盖下无解，不是完整机器人 C-space 不可达证明。此次碰撞区域覆盖由 12 个姿态区间构成，独立留出审核共 2,400 个样本、未检出碰撞；这仍是数值抽样证据，不是连续时间或区域全体安全证明。

## 解释与边界

这个实验说明了教授提到的现象：当 planner 只知道碰撞与底盘模式，而不知道“机械臂在该阶段必须留在房间内”时，GCS 可以返回一个数学上可行但违反人的任务直觉的解。加入全臂语义后，该语义改变了凸区域边界和可用的 GCS 边，而不是仅在轨迹出来后贴标签。

它是原始三维场景上的算法机制反例，但不是完整取放任务，也没有证明一般 GCS 成功率优于其他 planner。当前只优化二维仿射构型子集；碰撞及路径按离散状态审查。此前同一实验另有 8 项力驱动 SAP 执行记录，见原实验目录；**本轮新运行只重跑 GCS 规划与轨迹数值审核，没有重跑 SAP 动力学执行**。

## 复现与材料

- 实验实现：`src/arm_boundary_gcs.py`、`src/arm_boundary_semantics.py`
- 固定实验协议和 15 个规划结果：`outputs/results/arm_boundary_gcs_repro_20261011/`
- 可交互 Meshcat HTML：`outputs/results/grasp_fix_web/planar/arm_gcs_repro_20261011/index.html`（页面里的两个选项可切换 Base-only 和 Semantic GCS 轨迹）
- [对照页面截图](../outputs/results/arm_boundary_gcs_repro_20261011/web_preview.png)
- [原始实验详细报告](31_whole_arm_semantic_gcs.md)

复现命令：

```bash
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.arm_boundary_gcs
```

本次通过 `tests/test_arm_boundary_gcs.py` 和 `tests/test_arm_boundary_semantics.py`：6 项测试全部通过。复现源码与对应历史结果中的规划器、谓词及场景 fixture SHA-256 一致。
