# 项目工作总览：Drake、语义约束、GCS 与 C-space 连通性

更新时间：2026-10-10
项目：`/home/hudi/projects/convex`

本文汇总本项目从最初双房间抓放原型到近期门口取物与稀疏语义细化实验的主要工作、证据和未完成问题。它是导航文档，不替代每项实验的协议、冻结源码、原始记录与详细报告。数字引用的实验范围各不相同，不能跨协议直接比较。

图文 PDF 版本：[61_project_work_summary.pdf](61_project_work_summary.pdf)；LaTeX 源文件：[61_project_work_summary.tex](61_project_work_summary.tex)，截图素材位于 `reports/figures/project_summary/`。

## 研究目标与当前认识

项目研究如何把任务语义、机器人 C-space 和 GCS 运动规划结合起来，让语义任务不仅在符号层面顺序正确，还能细化成连续、无碰撞、可执行的机器人轨迹。

目前需要区分三个问题：

1. **语义是否定义正确**：例如机器人是否必须进入某个房间、手臂或整机是否可以越过边界、抓取前后允许哪些动作。
2. **语义允许的 C-space 是否有可行连续路径**：合法的起点和终点语义区域可能在当前 IRIS 凸区域覆盖图中断开。
3. **GCS 是否带来相对强基线的收益**：语义任务正确或 GCS 找到一条成功轨迹，并不能单独证明 GCS 比强非 GCS 方法有更高成功率或效率。

目前证据支持：语义约束会影响任务是否有效；IRIS/GCS 能在明确的区域图和任务模式下规划连续轨迹；稀疏端点区域断连时，可以按 GCS-Bridging 思路补入中间凸区域，在当前门口示例中得到完整轨迹。**目前没有证明 GCS 普遍比强的 Dijkstra/Portal + 连续优化基线成功率更高或更快。**

## 场景与系统演进

### 原始双房间完整取放任务

Drake 场景包含 Room 1、较小的 Room 2、门洞、移动底盘、7-DOF KUKA iiwa、Schunk WSG 夹爪、桌子、杯子、绿色放置箱、柜子、推车及干扰物体。原始任务是：从 Room 1 出发，底盘通过门洞进入 Room 2 并到达桌边，机械臂抓取杯子，再把杯子运回 Room 1 放入箱子。

早期语义序列包含 `base_in_room1 → base_in_room2 → near_pickup_table → cup_grasped → cup_in_dropoff_bin`。当时房间谓词主要检查**底盘参考点**，不是整台机器人几何体是否完全处于房间内。底盘导航和机械臂/携物操作分阶段规划；碰撞检查会约束机器人几何，但没有明确的整机“始终在房间内”语义不变量。

这段从初始状态进入取物区域的导航轨迹，在结构上类似教授最近提出的 motion bridge，但当时没有把它作为“连接初始配置与语义有效 C-space 区域”的独立研究问题来定义和比较。

### 门口伸手场景

根据教授对门口任务的讨论，后来构造了另一类任务：底盘停在 Room 1，手臂从门口伸进 Room 2 抓杯子，再把杯子撤回 Room 1。桌子与杯子反复调整到 Room 2 一侧，目标是直观展示“底盘在外、手臂伸入”的行为，并排除底盘穿门这一捷径。

当前 `scene_doorway_reach_semantic_v9.yaml` 系列配置和对应 trace 记录门平面、底盘、末端、手臂 link、杯子及抓取语义。此任务与教授最新提出的 green-room toy example **不是同一个任务定义**：当前门口任务要求底盘留在 Room 1、机械臂进入 Room 2；教授的 toy example 则是先让整台机器人进入绿房间，再要求之后的手臂操作始终在房间内。

## 已完成的技术工作

### 1. 场景、仿真和抓取可信度

- 用 Drake 建立双房间 3D 场景，使用 Meshcat 展示和回放。
- 建立底盘导航、机械臂运动学、碰撞检查、IRIS/ GCS 规划及独立语义 trace 的基本流程。
- 修复早期抓取模型中过度过滤杯子与夹爪碰撞几何的问题，增加抓取几何审核和双指接触检查。
- 对完整任务进行自由杯子 SAP 接触动力学执行。验证夹爪闭合时的双指接触、抬升、携带、放置和释放；设置夹爪张开的负对照。
- 仿真仍使用理想全驱动平面底盘，不是轮式底盘动力学或真实硬件验证。

详细报告：[05 仿真方法审查](05_simulation_method_review.md)、[06 抓取模型修复](06_grasp_model_repair.md)、[07 完整接触执行](07_full_contact_execution.md)。

### 2. 语义谓词、任务状态机与 Product 图

- 建立房间/区域谓词、底盘导航语义、near-pickup、抓取/携物/释放等任务状态和模式。
- 将任务顺序与凸区域图组合，限制规划器可使用的区域和动作顺序。
- 增加逐帧语义 trace、signed residual、抓放 guard、碰撞和任务顺序审核。
- 发现重要建模原则：区域名字或中心点满足谓词，不代表区域内所有配置都满足谓词；需要检查全区域语义或采用明确的 guard。
- 当前自然语言指令到谓词/任务图的映射仍主要由人工指定；没有完成通用 command-to-predicate mapping。

文献和设计背景见[报告 01](01_semantic_modeling_literature_review.md)、[报告 17](17_semantics_inside_gcs.md)、[报告 30–34](30_whole_arm_boundary_semantics.md)。

### 3. C-space、IRIS 与 GCS 规划

- 比较过二维圆盘/局部 bubble、姿态相关圆盘、二维/SE(2) 底盘导航区域、机械臂关节空间凸区域及携物 10D 配置空间区域。
- 使用 Drake IRIS-NP2 生成碰撞相关凸区域，并使用 Drake `GcsTrajectoryOptimization`/Graph of Convex Sets 与 Clarabel 求解区域图中的连续路径。
- 携物规划中将杯子几何纳入碰撞模型；按自由杯、抓取/附着、释放等不同模式构建不同碰撞场景和区域。
- 将抓取/放置操作的部分阶段纳入 mode-specific IRIS 区域和 GCS；仍有夹爪开合等动作阶段由预设参考动作驱动，不能笼统声称所有自由度和接触转换都在一个凸 GCS 中联合优化。
- IRIS-NP2 的有限抽样审计不是整个凸区域的严格无碰撞证书；轨迹有限采样也不是连续时间安全证明。

与 GCS 作用相关的指定论文包括 [Motion Planning around Obstacles with Convex Optimization](https://arxiv.org/abs/2205.04422) 和 [Growing Convex Collision-Free Regions in Configuration Space using Nonlinear Programming](https://arxiv.org/abs/2303.14737)。项目也实现了真正不调用 GCS 的离散区域搜索加 SOCP 对照，而不是把仍使用 GCS 的方法误称为非 GCS 基线。

### 4. 多任务评估、失败诊断与 GCS 对照

项目进行过固定任务集、在线 IRIS 修复、动作候选生成、携物规划、GCS 舍入及任务语义消融等实验。关键结果应按实验范围分别解读：

| 实验 | 主要观察 | 可支持的结论 |
|---|---|---|
| 冻结 40 任务评估（报告 44） | 分阶段 GCS 与 Product-GCS 各 5/40；加入在线 IRIS 的 Product-GCS 为 11/40 | 在线区域修复恢复了部分原先端点未覆盖的任务；动作原语缺失仍是主要瓶颈 |
| 统一在线修复（报告 46） | 一个评估中 40 个新任务由 12/40 提升至 16/40；28 个剩余失败在三种高度下都没有合格动作 | 增加动作/端点覆盖可恢复部分任务；不可把无 IK/动作候选说成 GCS 求解失败或物理不可行证明 |
| 语义与 GCS 机制对照（报告 47） | 40 个任务上 Product-GCS 与 Dijkstra + SOCP 都满足语义，预定接触子集均 12/12；忽略检查点的消融仅 12/40 参考满足、接触 5/12 | 语义对任务有效性很重要；该实验没有显示 GCS 相对任务感知非 GCS 基线的成功率优势 |
| 统一多目标对照（报告 49） | GCS 与 Dijkstra + SOCP 都规划通过 24/24，预选接触执行都为 6/6；非 GCS 方法更快 | 当前耦合难度不足以证明 GCS 成功率更优，不能弱化基线 |
| 新物理条件筛查（报告 50–51） | 不同门宽、推车和杯子摆放条件下，三种查询器的任务结果接近；共同区域筛查修复了部分有限区域碰撞反例 | 区域安全与场景泛化仍是问题；旧场景回归不能代替独立新场景评测 |

完整协议、分母、失败类别、置信区间和费用口径以各报告及输出目录中的 `protocol.json`/原始记录为准。不要把不同数据集的成功率拼成单一排行榜。

相关报告：[44 冻结任务评估](44_frozen_40_task_evaluation.md)、[46 统一在线修复](46_unified_online_delivery_evaluation.md)、[47 语义与 GCS 机制对照](47_semantics_gcs_mechanism_comparison.md)、[49 GCS 与 Portal/SOCP 诊断](49_global_portal_and_rounding_diagnosis.md)、[50–51 新物理条件和区域筛查](50_new_physical_conditions.md)。

### 5. 导航 C-space 覆盖修复

教授早先板书将重点分为底盘自由空间谓词、全臂 C-space 谓词及整机交互谓词。此前留出任务的大量失败发生在共享导航覆盖或候选准备阶段，而不是最终 GCS 查询阶段。项目据此诊断产品图路线、通道覆盖、门洞姿态以及固定朝向切片等问题，并把导航区域从固定朝向 XY 切片推进到 SE(2) 形式，纳入收臂姿态和门洞通行条件。

修复后，旧 16 场景开发回归中干净完成场景保持零回归；其中一些超时任务仍未判定，不能算成功或失败。报告 56 中的七维全臂覆盖质量、新独立场景评测和自动指令映射仍列为欠账。

详见[报告 53](53_bootstrap_navigation_three_layer_diagnosis.md)、[报告 54](54_weekly_report_se2_navigation.md)、[报告 56 周报](56_weekly_requirements_vs_progress.md)。

## 当前研究线：稀疏语义计划与 C-space 连通性

教授最近强调：用户给出的稀疏语义约束可能没有考虑 C-space 连通性；语义端点区域即使单独可行，也未必存在连接它们的连续轨迹。需要找到并加入缺失的中间语义或运动结构，再验证所得计划是否可以连续细化。

### 稀疏覆盖诊断

在门口 reach 阶段用 2、5、9、17 个 C-space guide samples 作区域生成种子。2 点和 9 点情形得到的凸区域图断连；5 点和 17 点情形连通并由 GCS 求解成功。这说明该实现的 cover repair 对输入密度并不单调。该例的碰撞自由空间存在通过有限插值审核的路径，因此观察到的是**当前凸覆盖图断连**，不是证明真实 C-space 拓扑断开。

报告：[58 稀疏语义连通性诊断](58_sparse_semantic_refinement_connectivity.md)。

### 依据 GCS-Bridging 的连接修复

按 [GCS-Bridging 预印本](https://arxiv.org/abs/2608.22326) 的 DirectLine / binary inflation 思路，在两个互不相连的端点 IRIS-NP2 区域之间做跨分量见证点检查，并沿通过碰撞检查的路径扩张中间区域：

- 初始为 2 个区域、2 个分量；DirectLine 通过碰撞检查，LocalRRTC 因此未运行；
- 3 次区域扩张后图连成一个分量，最终图 5 个区域；最短连接链包含 4 个区域；
- 局部 reach GCS 轨迹通过 200 段碰撞审核；对应报告中 1,000 个 IRIS 留出样本未检出碰撞。

之后把连通性修复接入完整门口任务语义图，用**一次全局 9D GCS 求解**选择底盘导航、手臂 reach、双指抓取动作和携物撤回轨迹，并运行 SAP 自由杯接触仿真：底盘到门平面最小间隙 0.146 m；末端进入 Room 2 最深 0.499 m；双指接触抓取并把杯子带回 Room 1；非预期接触 0；142 个保存轨迹构型的独立碰撞审核均通过。

重要限制：attached-cup IRIS 区域的 1,600 个留出样本中检出 3 个碰撞构型。所选轨迹与 SAP 执行通过，并不认证这些凸区域中所有点都安全。该 full-task 场景仍是“底盘留在 Room 1、手臂伸入 Room 2”，**不是教授刚提出的 whole-robot-inside green-room toy task**。

报告：[59 局部 GCS-Bridging 复现](59_gcs_bridging_paper_reproduction.md)、[60 完整语义 GCS 与 SAP 验证](60_unified_semantic_gcs_bridging_physical_validation.md)。3D 页面和机器结果位于 `outputs/results/doorway_arm_reach_semantic_v13/`。

## 教授最新提出的 toy example 与下一步解释

教授提出的最小语义条件是：

1. `whole-robot-inside(green room)`：在开始机械臂操作前，整台机器人进入绿房间；
2. `whole-motion-inside(green room)`：在机械臂执行 reach/grasp 的整段运动中，整台机器人始终处于绿房间内。

初始状态在语义有效 C-space 区域之外，因此还要规划一段从初始状态进入有效区域的 motion bridge。项目此前的原始取放流程确实已经有“底盘从 Room 1 经门口进入 Room 2，再停在桌边后由机械臂抓取”的相似分阶段结构；但旧谓词只按底盘参考点判断是否进入 Room 2，且没有显式保证 whole-robot / whole-motion 语义条件。因此是**结构相似，形式语义未完成**。

下一步应优先按教授的新例子建立或恢复相应的全机器人入室设置，并明确：

1. 绿房间几何边界与安全余量；
2. “整台机器人在房间内”的几何判定，不能只检查 base 原点或末端；
3. 底盘移动阶段手臂采用何种固定收拢姿态；
4. 机械臂操作阶段整机语义不变量如何落实到允许的 C-space 区域；
5. 从初始配置到该语义区域的 collision-free motion bridge；
6. 无桥接/语义约束、加入 bridge 后的可复现对照；
7. bridge 与操作路径的连续连接、完整 GCS 轨迹及物理/离散审核。

需要先把这个 green-room 问题与门口“底盘在外、手臂伸入”的已有演示区分开。不要将两个不同任务的语义谓词混用，也不要把“GCS 图无路径”直接解释为真实运动不可达。

## 系统与代码入口

| 功能 | 主要文件 |
|---|---|
| 场景、机器人和碰撞几何 | `src/build_scene.py`、`src/scene_spec.py`、`configs/scene.yaml` |
| 语义谓词、residual、任务监测 | `src/semantic_predicates.py`、`src/task_semantics.py`、`src/semantic_monitor.py` |
| 底盘 GCS 与区域 | `src/gcs_base_planner.py`、`src/geometry_outbound_gcs.py` |
| whole-body / 携物 GCS 和 IRIS | `src/gcs_whole_body_planner.py`、`src/iris_region_generator.py`、`src/manipulation_gcs.py` |
| 完整任务规划与执行 | `src/animate_delivery_demo.py`、`src/execute_delivery_dynamics.py` |
| 门口 GCS-Bridging 实验 | `src/experiment_gcs_bridging.py`、`src/animate_unified_doorway_gcs.py` |
| 失败与机制分析 | `reports/` 中对应编号报告；实验原始结果在 `outputs/results/` |

项目有很多历史输出和报告。复现实验应读取对应目录中的协议、场景快照和 `source_snapshot`（如果有），不能默认当前源码与历史运行源码一致；不要覆盖已有结果目录。

## 目前不能声称的结论

- 不能说 GCS 已经证明比强非 GCS 基线有普遍更高成功率或更快速度。
- 不能说“使用 IRIS-NP2”就得到整个凸区域严格无碰撞保证。
- 不能说离散轨迹样本无碰撞等于连续时间安全证明。
- 不能把有限区域图断连等同于真实 C-space 不连通或任务物理不可行。
- 不能说自然语言指令已经自动映射到语义谓词和任务图。
- 不能把当前 base-outside doorway task 当成教授新 green-room toy example 已完成。

## 关键报告索引

- [01 语义建模文献调研](01_semantic_modeling_literature_review.md)
- [06 抓取模型修复](06_grasp_model_repair.md)、[07 完整接触执行](07_full_contact_execution.md)
- [17 语义进入 GCS](17_semantics_inside_gcs.md)、[30–36 全臂谓词与 C-space](30_whole_arm_boundary_semantics.md)
- [43–46 完整任务、冻结评估和在线修复](43_full_task_development_baselines.md)
- [47 语义/GCS 机制对照](47_semantics_gcs_mechanism_comparison.md)、[49 GCS 求解器诊断](49_global_portal_and_rounding_diagnosis.md)
- [52 项目历史与交接](52_task_history_and_handoff.md)、[56 教授要求与进度](56_weekly_requirements_vs_progress.md)
- [57 门口场景与教授反馈](57_professor_feedback_scenario_semantics.md)
- [58 稀疏语义连通性](58_sparse_semantic_refinement_connectivity.md)、[59 GCS-Bridging 局部复现](59_gcs_bridging_paper_reproduction.md)、[60 完整 GCS 与 SAP 验证](60_unified_semantic_gcs_bridging_physical_validation.md)
