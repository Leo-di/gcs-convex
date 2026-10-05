# 语义建模与 GCS：第一轮文献调研及实现建议

调研日期：2026-09-29。代码基线：`d51334c0b39bf170a5c0e81e3ffe07968d81c13f`。

本文区分文献结论、代码观察和设计建议。重点核读 LTL-GCS 的问题定义与区域标签假设、STL-GCS 的联合状态构造；其余工作为论文摘要、方法概览和官方代码说明层面的筛选。尚未复现外部基线，也未穷尽相关文献，不能据此宣称创新性。

## 1. 建议采用什么框架

建议采用“有限任务规格 + 显式操作模式 + 几何动作条件 + 模式相关凸区域”。任务内容由我们定义，表达和验证方法参考研究。

短期保留手工任务自动机，明确其接受条件和失败状态；把有限轨迹上的任务规格写清楚，再引入 LTLf 编译器。移动、抓取、携带、释放的物理模式应与任务进度分别记录，不能只给空间区域贴一个名称就认定操作可执行。

LTLf 适合描述这次有明确终点的交付任务。它是在有限轨迹上解释的时序逻辑，不能不加说明地沿用无限轨迹 LTL 的结尾语义。理论入口：[De Giacomo & Vardi, IJCAI 2013](https://www.cs.rice.edu/~vardi/papers/ijcai13.pdf)。这里选择 LTLf 是针对本项目的建议，不意味着必须立即替换现有自动机。

## 2. 优先阅读的研究

| 优先级 | 文献与来源 | 可借鉴内容 | 对本项目的判断 |
|---|---|---|---|
| P0 | Kurtz & Lin, **Temporal Logic Motion Planning with Convex Optimization via Graphs of Convex Sets**, T-RO 2023；[论文](https://arxiv.org/pdf/2301.07773)、[作者代码](https://github.com/vincekurtz/ltl_gcs) | 自动机与凸区域结合，使用 GCS 规划连续轨迹 | 最直接基线，product + GCS 本身已有先例 |
| P0 | Garrett, Lozano-Pérez & Kaelbling, **PDDLStream**, ICAPS 2020；[论文入口](https://ojs.aaai.org/index.php/ICAPS/article/view/6739)、[作者代码](https://github.com/caelan/pddlstream) | 用黑盒采样过程连接符号规划与连续几何关系 | 借鉴 IK、抓取和路径可行性提供几何见证的接口，不必立即移植整套规划器 |
| P0 | De Giacomo & Vardi, **Linear Temporal Logic and Linear Dynamic Logic on Finite Traces**, IJCAI 2013；[论文](https://www.cs.rice.edu/~vardi/papers/ijcai13.pdf) | 有限执行轨迹上的规格语义 | 明确结束条件、动作事件与持续状态的区别 |
| P1 | Graesdal et al., **Towards Tight Convex Relaxations for Contact-Rich Manipulation**, RSS 2024；[官方论文页](https://www.roboticsproceedings.org/rss20/p132.html) | 用 GCS 表达接触模式序列及模式内的连续问题，实验对象包含平面推动 | “模式 + GCS”也不是空白；其接触动力学不能直接当作本项目 attachment 的证明 |
| P1 | Wei, Luo & Liu, **Hierarchical Temporal Logic Task and Motion Planning for Multi-Robot Systems**, RSS 2025；[官方论文页](https://www.roboticsproceedings.org/rss21/p099.html) | 分层时序任务、任务分配与凸区域运动规划的 product graph | 多机器人扩展的重要近邻；目前先解决单机器人语义—几何一致性 |
| P1 | Chen et al., **Signal Temporal Logic Motion Planning via Graphs of Convex Sets**, 2026；[预印本全文](https://arxiv.org/html/2605.23240v1) | timed automaton、区域约束及连续时间轨迹联合建模 | 出现截止时间、持续时间要求时再优先采用；本文未核实正式录用状态 |
| P2 | You, Luna & Summers, **A Framework for Motion Planning with Temporal Logic Precedence Specifications via Augmented Graphs of Convex Sets**, 2026；[预印本全文](https://arxiv.org/html/2606.00842v1) | 用分层图编码钥匙—门等前序约束 | “先抓后放”这种先后逻辑不足以单独构成新方法；未核实正式录用状态 |
| P2 | Lin et al., **Towards Tighter Convex Relaxations of Mixed-Integer Programs: Leveraging Logic Network Flow for Task and Motion Planning**；[2025 首稿、2026 v2](https://arxiv.org/html/2509.24235v2) | 将逻辑约束放入网络流结构，研究松弛紧致性 | 若教授希望改进优化算法，这是值得进一步核读的方向；暂不移植 |
| P2 | Natarajan et al., **Implicit Graph Search for Planning on Graphs of Convex Sets**, RSS 2024；[官方论文页](https://www.roboticsproceedings.org/rss20/p113.html) | 搜索与局部轨迹优化交错，避免一次性处理整个大图 | 图规模增大后的效率基线，不是当前语义定义缺口的替代方案 |

补充可靠性依据：[Drake IRIS 官方文档](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html)。IRIS-NP2 的概率保证针对区域内碰撞体积比例，而非区域每一点严格无碰撞，也不直接给出优化轨迹的碰撞概率。

## 3. 最直接文献带来的约束

Kurtz & Lin 的论文 III-B 明确规定：区域标签适用于该区域中的每个配置。其方法把自动机与连续几何放进 GCS；我们的代码则先做离散路径选择，再限制连续 GCS 的可用区域，不能直接继承该论文的最优性或正确性结论。[原文](https://arxiv.org/pdf/2301.07773)

2026 STL-GCS 的 Definition 4 同样要求区域包含于对应任务状态允许的集合，不能仅凭区域中心或名字判断兼容性。[原文](https://arxiv.org/html/2605.23240v1)

由此给本项目提出两类不同的接口：

- **持续区域条件**：在某段轨迹所在区域内，要求谓词处处成立。
- **动作切换条件**：只在抓取/释放发生的切换配置成立，但必须给出可达的几何见证以及执行后的状态更新。

## 4. 当前代码中的实质缺口

### 4.1 区域标签不等于几何事实：已验证反例

`configs/scene.yaml` 将 `room2_pickup_lane` 标记为 `near_pickup_table`。该区域 XY 边界为 `[0.25, -0.45]` 到 `[1.35, 0.05]`，桌中心 XY 为 `[1.60, 0.65]`，运行时的 near 阈值为 1 米。

区域内角点 `(0.25, -0.45)` 到桌中心的距离为：

```text
sqrt((0.25 - 1.60)^2 + (-0.45 - 0.65)^2) = 1.7414074767 m > 1 m
```

四角检查中只有 `(1.35, 0.05)` 满足 near。这个确定性反例足以否定整个区域的统一 near 标签；并不意味着演示中固定的实际抓取点不满足 near。

建议把此区域保留为移动通道，并为 pick 边设置几何 guard，或建立经过验证的独立抓取子区域。仅检查 near 仍不足以证明 IK、夹爪对齐和接近路径可行。

### 4.2 顺序见证不是完整语义监测

`src/animate_delivery_demo.py::_verify_ordered_events` 顺序寻找真值帧。它不会检查提前发生的释放、搬运中断、全部模式不变量或非法模式切换。`cup_grasped` 由帧生成代码赋值，不是独立的抓取状态测量。

下一版 trace 应额外保存对象位姿、夹爪状态、attachment 关系、任务状态、物理模式和动作事件。监测器根据这些数据独立重算条件；不能把 phase 名称直接当成成功依据。

### 4.3 先离散搜索，后优化连续轨迹

`src/semantic_product_planner.py` 使用 XY 区域重叠及区域中心间距离建图；移动保持 mode，动作边检查手工标签。`animate_delivery_demo.py` 再把所选模式路径的区域交给 GCS。

因此当前是分层规划原型，不是对所有任务顺序与连续轨迹做联合全局优化。若所选离散路线的几何优化失败，尚没有在这里看到自动切换到其他任务路线的闭环。离散 XY 邻接也不自动保证姿态、关节或切换动作可行。

### 4.4 区域和轨迹的证据级别不同

本机先前实测：433 帧检查通过，14 个 carrying 区域，cover 审计 3/2800 碰撞。IRIS 的体积比例保证不能推出最优轨迹必然避开碰撞子集；轨迹被优化器选择，也不是区域中的均匀随机样本。[保证的定义](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html)

目前不能把区域抽样审计、离散轨迹检查、连续时间安全证明混用。也不要把“有限凸覆盖中无解”说成“原机器人任务不可能完成”。

## 5. 推荐的最小建模方案（我们的设计建议）

### 状态与模式

完整监测状态至少记录 `z=(q, gripper, X_WC, attachment, task_state, mode)`，其中 q 为底盘和机械臂配置。携带时显式存储固定抓取变换 `X_GC`，使杯位姿满足 `X_WC = X_WG(q) X_GC`；这仍然是运动学模型。

区分任务进度 `s` 与物理模式 `m`。第一版可以一一对应，接口上仍应分开。后续允许放下后重抓时，这两者会明显不同。

| 元素 | 判定与证据 | 使用位置 |
|---|---|---|
| `base_in_room2` | 底盘参考点落在房间内；不要解释成整个机器人都在房间内 | 状态标签 |
| `near_pickup` | `1.0 - norm(base_xy - table_xy) >= 0` | 抓取 guard 的必要条件之一 |
| `grasp_feasible` | 指定对象与抓取变换、IK 解、容差、接近段碰撞验证 | 抓取动作的几何见证 |
| `attached(cup, gripper)` | 离散 attachment 关系与位姿一致性共同检查 | carrying 模式不变量 |
| `cup_inside_bin` | 杯子整体与箱体内部的几何关系，必要时考虑倾斜姿态 | release guard |
| `supported` | 几何支撑接触/高度容差；不冒充力学稳定性 | release guard 和交付终态 |
| `collision_clear` | 模式对应碰撞模型下的 signed-distance residual | 各运动阶段约束 |

建议动作表：

| 动作 | 前置条件 | 状态更新 |
|---|---|---|
| attach/pick | 未附着、目标身份正确、near、抓取对齐及接近轨迹有效 | 记录 attachment 与 X_GC，进入 carrying |
| carry | attached、模式碰撞模型有效 | 持续保持 attachment，杯位姿随夹爪更新 |
| release/place | attached、杯子整体在箱内、几何支撑成立 | 解除 attachment，记录杯子的世界位姿 |
| retreat/finish | 杯子留在目标箱且未附着，撤回段有效 | 进入任务接受状态 |

这些条件是本项目建议，借鉴 PDDLStream 的“符号关系需要连续几何见证”思想，并不声称是 PDDLStream 的原有任务定义。[来源](https://ojs.aaai.org/index.php/ICAPS/article/view/6739)

### 规格与监测

先用任务自动机严格定义合法动作序列和唯一接受条件；附加有限轨迹监测规则，例如：

```text
G(carrying -> attached)
G(pick_event -> near_pickup AND grasp_feasible)
G(release_event -> inside_bin AND supported)
F(delivered)
```

以上只是规则片段，不是完整任务公式。约定动作事件在切换前状态检查，attachment 更新在切换后状态检查；自动机负责先抓后放、禁止多余释放，以及 delivered 的进入条件。否则 release 同一帧中的 attached 真值会产生歧义。终态必须是已释放并完成规定撤回，不能仅凭杯子进入箱内就结束。

第一版监测明确针对采样帧；涉及区间持续约束时另外记录是否有连续段证据。

### 区域语义与 guard 几何

对模式 m 中的区域 C，只有证明 `C ⊆ {q : p(q,m)=true}` 才可赋予全域为真的标签。跨越边界则标记 mixed/unknown，进一步切分或转为动作 guard。未知不等于假；若逻辑使用否定标签，也必须证明该区域内谓词处处为假。

仿射谓词可通过线性规划检查最坏余量。非线性 IK、碰撞和姿态条件不能只检查多面体顶点，更不能把局部优化得到的最小值当作全局下界；可先用明确标注证据级别的保守内近似或有限动作见证。

未来扩展节点 `(C_i^m, m, s)`。模式切换不只是两个区域相交，还要满足切换 guard 和 reset 关系。抓取接触等约束可能形成低维集合，应使用固定切换配置或局部参数化，不能假定都可被普通满维 IRIS 区域精确表示。把非凸 guard 原样加进 GCS 也不会自动保持凸性。

## 6. 实验设计与实施顺序

1. **统一谓词定义**：生成 trace 与规划 guards 共用定义，执行监测从原始状态重新计算；保存阈值、单位和几何对象。
2. **修复标签与动作条件**：先处理 pickup lane 反例，新增释放条件、attachment 连续性与失败报告。
3. **反例测试**：远离桌子抓取、搬运时解除 attachment、箱外释放、错误对象、先释放后出现合法序列、未完成释放就终止。应报告首个违反条件的帧和数值依据。
4. **补足模式相关几何**：优先检查抓取/释放两侧的切换配置，再扩展各模式覆盖，保留数值验证与证明的区别。
5. **做公平对照**：相同任务、场景、几何模型与预算，对比固定正确任务顺序 + GCS、现有标签 product、guard 增强 product；再复现 LTL-GCS 的简化等价任务。纯几何 GCS 仅作不使用任务约束的消融，不作为同等任务求解器宣传。
6. **增加真实决策分支**：当前单杯单箱的顺序几乎固定，难以体现任务规划价值。加入多个合法抓取候选、两个允许的放置箱或暂存后重抓任务，比较离散路线与连续代价的协调。

指标：任务成功率、语义违规、碰撞、最小几何余量、区域标签反例数、IK/区域生成/图构造/GCS/验证分项时间、失败原因。先固定可复现种子调试，再用独立种子重复实验；审计样本与生成样本分离。区分“监测器检测出违规”与“规划器已避免违规”。

若使用实际截止时间，必须先让动画时间、GCS 时间参数和执行时间一致，再引入 STL 时间约束；帧编号本身不能替代物理时间。

## 7. 可能的研究问题与不能声称的贡献

候选问题：面对 attachment 改变的移动操作任务，怎样构造带有明确语义证据的模式相关凸区域与切换连接，减少虚假的符号可行计划，并控制构图和求解开销？

这是有实验价值的方向判断，不是已确认的新颖性结论。优先检验“标签/guard 校验能否降低执行违规及无效几何求解”，之后才考虑按需区域细化、复用及联合规划。

目前不能把以下内容单独作为新算法：自动机 × 区域 product、LTL/STL + GCS、接触/操作模式 + GCS、使用 IRIS 自动生成区域。上述文献均存在直接或相近先例。

下一份可交付成果建议是：谓词与动作规格文件、独立语义监测结果、pickup 标签反例及修复对比、包含分支选择的第一版对照表。无需先接大语言模型。
