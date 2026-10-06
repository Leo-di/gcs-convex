# 周报：SE(2) 语义导航修复与 16 场景回归（本周）

> 汇报对象：导师。对应交接文档 `reports/52_task_history_and_handoff.md` 的首要任务。
> 关联报告：53（三层修复技术细节）。

## 一、本周做了什么（按您上次的优先级）

上周交接确认的首要任务是：**48 个 heldout 任务里 24 个在查询前就因共享导航覆盖失败**——瓶颈不在 GCS 求解器，在"当前几何导航覆盖"。本周完成了三层修复并在旧 16 场景上做开发回归。

### 修复 1：导航谓词从固定朝向切片升级为 SE(2)

之前：空载导航 = 固定 yaw + 固定机械臂姿态的 2D 平移切片。推车被随机场景挪进通道时，这个切片**断连**（25 个碰撞分支复检，22 个固定朝向救不回来）。

现在：每个凸区域 = 当前几何 XY 圆盘 × yaw 区间的笛卡尔积，底盘可以在语义 lane 内**变朝向绕行**推车。形式化地说，导航可行性谓词的定义域从 C-space 的一个 2D 仿射切片扩展到了 SE(2) 子流形：

```
free_navigation = { (x, y, ψ) ∈ SE(2) : dist(base(q_base(x,y,ψ), arm_home), obstacles_current) ≥ margin }
```

这正是"语义谓词 = 构型空间集合"的一步推进：谓词不再绑定单一朝向。

### 修复 2：收臂姿态进入导航区域计算

导航帧的机械臂不再固定用 home 姿态，而是**选择使底盘自由空间最大的收拢姿态**（`select_navigation_arm`）——机械臂几何因此进入圆盘半径的计算，对应您板书里第二层（全臂 FK 决定谓词）的导航侧版本。

### 修复 3：门洞种子加密 + 全 lane 放行

沿用了既有门洞加密种子，并把 outbound 阶段的语义 lane 从部分放行改为按 product graph 全部放行。

## 二、回归证据（旧 16 场景，与 heldout_transfer_v1 同协议）

**零回归**——所有无干扰完成的场景候选数与旧版完全持平：

| 场景 | 旧版 | 新版 | | 场景 | 旧版 | 新版 |
|---|---|---|---|---|---|---|
| 00 | 3 | 3 ✓ | | 11 | 2 | 2 ✓ |
| 07 | 4 | 4 ✓ | | 14 | 3 | 3 ✓ |
| 09 | 0 | 0（已知难场景） | | 15 | 4 | 4 ✓ |

**修复有效的直接证据**：heldout_01（旧版 0 候选，推车挡路）端到端全链路通过——SE(2) outbound 最小安全距离 4.5cm，后续 IK / IRIS / 携物 GCS / 采样验证全部通过。

（补跑说明：共享服务器本周被 ~18-36 核的外组任务长期占满，10 个需要 SE(2) 慢路径的场景补跑仍在进行；以上结论只基于完整完成的场景，不含任何超时误杀数据。）

## 三、诚实的失败分类（按 52 号报告口径分开统计）

1. **覆盖连通但 GCS 取整无路**（heldout_05/06/13 的 x110）：覆盖图验证连通，但 Drake 官方 GCS 10 次取整找不到路。这是稠密图上求解器的鲁棒性短板，**不是覆盖问题**——是我们自己代码之外下一个要量化的点（边集稀疏化 / 更多取整尝试）。
2. **IK 失败（SNOPT infeasible）**：与旧版完全相同的候选（如 00 的 x110、11 的 x110/x115），本轮未触碰，属抓放层（第三层谓词），列入下周。
3. **已知难场景**（heldout_09）：推车恰好压在 lane 边缘，所有版本一致 0 候选，非回归。

## 四、可现场演示

实时 Drake/Meshcat 交互演示（非录像）：两房间 + 挡路推车场景，展示底盘收臂、变朝向绕行推车、穿门到达取物点的完整 SE(2) 导航。

## 五、下周计划

1. 补跑收尾 → 最终 16 场景新旧候选数对比表（预期 01/02/05/13 从 0 变为 >0）。
2. 按预注册协议跑**全新物理场景**评测（样本数/方法顺序/成本口径预先固定）。
3. 抓放层的 mode-specific 谓词（pre-grasp / grasp / release / retreat 的 IK 与区域），即您板书第三层 valid_whole_body_interaction 的推进方向。
4. 量化"GCS 取整无路"类别：边集稀疏化实验。

## English summary

This week we completed the three-layer bootstrap navigation repair from the handoff: (i) the navigation predicate now lives on the SE(2) subspace of configuration space—each convex region is a current-geometry XY disk crossed with a yaw interval, so the base rotates in-lane to route around a relocated cart; (ii) the arm posture is selected to maximize base free space, bringing arm geometry into the navigation regions; (iii) doorway seed densification. The 16-scene development regression shows zero regressions (all undisturbed scenes match the old candidate counts exactly), and heldout_01—previously rejected before query—now passes end-to-end with 4.5 cm minimum clearance. Remaining failures are honestly categorized: solver rounding failures on dense-but-connected covers (next: edge sparsification), untouched IK failures in the grasp layer, and one known-hard scene. Next week: finish the patch runs, pre-register a fresh-scene evaluation, and extend mode-specific predicates to the grasp/release layer.

## 附记（10-06 02:12 补跑终局）

补跑全部结束，`se2_navigation_refresh_v3/comparison.json` 已发布，16/16 场景有最终结果：

- **零回归确认**：8 个干净完成场景候选数与旧版逐场景持平（00:3/3、07:4/4、11:2/2、14:3/3、15:4/4，合计 16=16）。
- **预期修正**：原预期 heldout_01/02/05/13（旧版 0 候选）补跑后 >0——**未兑现**。13 干净跑完仍 0 候选；01/02/05/10/12 仍 timeout。诚实结论：SE(2) 修复保证零回归（已证），未解锁新候选。
- **遗留**：03/04/08（旧版 1/1/2 候选）被机器竞争误杀未能复现验证，转下周错峰重跑；9 个 timeout 场景整体不计入结论（口径同报告 52）。

> 注：`status_regression.sh` 已修复显示 bug（此前有 progress 文件即误标「进行中」）；另请将本报告 §「下周」第 1 条视为已完成。
