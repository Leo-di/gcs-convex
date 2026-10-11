# 项目已完成工作与教授当前期望

整理日期：2026-10-11
项目：`/home/hudi/projects/convex`

本文把项目里已经完成的场景、语义建模、GCS/IRIS 实验和物理执行，与教授在 Slack、手稿和讨论中提出的目标放在一起对照。重点是避免把不同任务的场景和结论混为一谈。

## 先说结论

教授最新描述的 green-room toy example，**可以沿用我们早期双房间取杯任务的大体布局和动作顺序**：机器人从 Room 1 出发，底盘带着收起的手臂进入 Room 2，到桌边后由机械臂抓杯。现在已有 green-room 专项配置、whole-robot 房间谓词以及分阶段 GCS 原型，并且保存了一次通过的离散参考轨迹审核；所以 green-room 方向已从“待实现”推进到“有初步实现和数值结果”。

但“场景看起来类似”不等于“之前的实现已经满足要求”。旧版主要用底盘参考点判断房间位置；教授现在要明确表达并检查：

1. 导航结束、机械臂开始操作前，**整个机器人已经在目标房间内**；
2. **机械臂操作的整段运动中，整个机器人始终留在目标房间内**；
3. 初始状态在房间外，语义约束本身不会自动给出从初始状态到合法操作区域的连续路径，因此要检查或构造一条无碰撞的 **motion bridge**。

我们已有相似的分阶段取杯任务、语义 GCS 机制实验、稀疏 C-space 连通性诊断和 GCS-Bridging 复现。新建的 green-room 原型已将整机房间语义用于操作区域，并由底盘 GCS 从房间外到达整机位于房间内的停靠点；不过它是分阶段的 guide-seeded 规划，未做本轮 SAP 动力学执行，凸区域留出碰撞抽检也检出反例。因此当前最准确的状态是：**教授的 toy example 已有可运行原型和连续参考轨迹，仍需加强碰撞区域、动力学执行及稀疏语义自动补全方面的验证。**

## 教授希望解决的研究问题

教授最初提出的研究动机是：用户给出的语义计划可能很稀疏，也可能没有考虑 C-space 的连通性。每个语义状态各自看起来可行，不代表它们之间存在连续、无碰撞的轨迹。如果语义允许的构型区域在当前凸区域图中断开，就需要找到并加入缺失的过渡运动结构。

在他给出的最简单取杯例子中，任务分成两个运动规划阶段：

### 阶段 A：底盘导航到桌边

- 机械臂采用固定的收起姿态，主要由移动底盘导航。
- 目标条件是 `whole-robot-inside(green room)`：到达操作位置、开始机械臂操作前，整个机器人应在绿房间内。
- “整个机器人”不能只用底盘中心点代替；要按机器人碰撞几何或明确的机器人包络检查底盘、机械臂和夹爪。

### 阶段 B：机械臂抓取

- 底盘停在阶段 A 的导航终点，机械臂到杯子处完成 reach/grasp。
- 操作整段轨迹满足 `whole-motion-inside(green room)`。教授在解释中说的是机械臂 reach/grasp 过程中留在房间内；为与谓词名一致，建议实现和汇报时检查机器人整体在轨迹上始终处于房间内，而不只检查末端执行器终点。
- 杯子是否也被纳入“whole robot inside”的几何集合，教授原文没有说明，应在实现定义中明确标注，必要时当面确认。

### Motion bridge：连接初始状态与语义合法状态

初始构型 `q₀` 在房间外，因此它不属于满足 `whole-robot-inside(green room)` 的配置集合。需要一段连接运动：

```text
q₀（房间外）
  → 无碰撞地穿过门口，机械臂保持收起
  → q_nav（整个机器人在绿房间内，靠近桌子）
  → 固定底盘，机械臂在房间内 reach / grasp
```

Bridge 在进入房间的过程中当然会跨越房间边界；教授并不是要求从 `q₀` 开始整条轨迹都位于绿房间。语义边界条件的阶段含义应明确：导航阶段以“整个机器人进入房间”作为目标，抓取阶段再施加“运动全程在房间内”的不变量。

教授的重点还包括理解 GCS 如何处理这些条件：自然语言谓词需要落到配置空间集合/区域与图的连通关系上；GCS 在给定的凸区域和可连接边上选择连续路径。教授表示不清楚当前 Drake GCS 的具体用法，因此需要向他解释实际实现，而不能只说“用了 GCS”。

## 我们做过的工作

| 工作线 | 已完成内容 | 与教授当前目标的关系 |
|---|---|---|
| 原始双房间 3D 场景 | Drake + Meshcat 场景，移动底盘、7-DOF KUKA iiwa、Schunk 夹爪、门洞、桌子、杯子、放置箱及障碍物。旧完整任务是底盘进入 Room 2、抓杯、携杯回 Room 1 并放入箱子。 | 布局和动作顺序与教授的 toy example 大体接近；旧房间语义主要看底盘参考点，尚未证明整台机器人都在房间内。Room 2 是否足够容纳全机器人和操作过程，还需几何审核。 |
| 语义谓词和 Product graph | 有房间、区域、抓取/携物/释放等谓词和任务状态；把语义模式与凸区域组合成 Product 图；记录逐帧语义 trace、residual、碰撞和动作顺序。 | 证明了语义可以影响规划，但早期语义定义不能直接当作教授最新两个 whole-robot 谓词已经完成。 |
| C-space / IRIS / GCS | 使用 Drake IRIS-NP2 生成碰撞相关凸区域，以 Drake GCS / GcsTrajectoryOptimization 和 Clarabel 求连续路径；尝试底盘 XY、SE(2)、机械臂及携物 C-space。 | 具备实现基础。IRIS 区域和轨迹的有限抽样审核不是形式化的整区域或连续时间安全证明。 |
| 真实接触抓取 | 修复过杯子与夹爪碰撞过滤；自由杯 SAP 动力学执行验证双指接触、抬升、携带和放置，不靠重置杯子位姿或固定 attachment 完成物理抓取。 | 支持真实接触仿真，但教授当前 toy question 的核心是语义到连续轨迹的细化，不是优先扩展抓取物理。 |
| 门口伸手场景 | 另建任务：底盘留在 Room 1，机械臂穿门进入 Room 2 抓杯，再带杯撤回。做过 3D 动画、语义 trace、门框碰撞检查和 SAP 双指抓取执行。 | 这是“base 在外、arm 进入”的另一个任务，语义方向与 green-room toy example 不同。不能拿它直接回答“整机进入房间后在房间内操作”。 |
| Green-room whole-robot 原型 | 新增 `scene_green_room.yaml`、whole-robot 几何包含 oracle 与 staged GCS：底盘收臂导航、房间内 7D arm reach、7D lift；整机 collision geometry 参与房间边界检查，杯子另行检查。`green_room_v1` 的 550 帧参考轨迹在 1,655 个插值状态通过语义/碰撞/抓取几何审核。 | 这是目前最贴近教授 toy example 的实现。一次 GCS reference audit 通过，但本轮未做 SAP 动力学；reach 区域独立留出碰撞抽检为 2/920，故还不能报告为物理执行通过或整个凸区域安全。 |
| 稀疏语义与覆盖连通性 | reach 阶段用 2、5、9、17 个 guide samples 生成 IRIS 区域；2 和 9 个样本的当前覆盖图断连，5 和 17 个样本的图连通并由 GCS 求解。 | 直接展示了稀疏 C-space 结构可能造成图断连；这些是凸覆盖图结果，不是证明真实自由 C-space 拓扑断开。 |
| GCS-Bridging 复现 | 按论文 DirectLine / binary inflation 思路连接分离区域，补入中间 IRIS 区域；随后接入门口完整任务图，由一次全局 GCS 选择导航、reach、抓取和携物撤回，并做 SAP 执行。 | 展示了在门口任务中的连通性修复流程，但该任务仍是“底盘留 Room 1、手臂进 Room 2”，不等同于 green-room 任务。 |
| 无全臂语义反例 | 在同一场景下对照只保留底盘模式和碰撞约束的 GCS，与全臂语义 GCS。4 个正常条件中，基线均找到碰撞自由但违反全臂房间语义的轨迹；语义 GCS 均通过。 | 回答“几何可行不等于语义正确”，但不是 motion bridge 任务的完整实现。此实验的机器人状态写成 10D，实际限制在二维仿射姿态族上。 |
| GCS 基线评估 | 多组任务对比 Product-GCS 与离散区域搜索 + SOCP 等方法。已有结果中，部分配对集双方成功率相同，非 GCS 基线更快；有些语义消融显著损害任务有效性。 | 支持语义建模的重要性；目前**不能**声称 GCS 普遍比强非 GCS 方法有更高成功率或更快。 |
| 汇报材料 | 已有阶段总结、实验报告、LaTeX 源文件与多场景截图组成的 9 页 PDF。 | 汇总证据可供汇报，但不能替代 green-room 专项验证。 |

## 需要区分的三种场景

| 场景 | 底盘行为 | 手臂行为 | 是否是教授最新 toy example |
|---|---|---|---|
| 原始双房间完整取放 | 从 Room 1 穿门进入 Room 2，停在桌边；后来带杯返回 Room 1 | 在 Room 2 抓杯 | **布局和动作结构最接近**，但旧语义主要按底盘位置，未明确 whole-robot 终点/whole-motion 不变量与 bridge 定义。 |
| 门口伸手 | 底盘始终在 Room 1 | 穿门进入 Room 2 抓杯并撤回 | 否。它刻画 base-outside / arm-inside，和教授现在的 green-room 要求相反。 |
| 教授 green-room toy example | 底盘带收起手臂进入绿房间，导航阶段结束时机器人整体在内 | 底盘停住，机械臂在房间内抓杯，整段操作时机器人整体不越界 | **这是当前应明确实现和解释的任务。**可以复用原始双房间布局，但需要重新定义语义并审核房间尺寸、门洞、姿态和整机几何。 |

因此，回答“是不是之前的场景”：**宏观上很可能是同一类双房间取杯场景；技术上不是旧实现已经做完。**复用场景前，先确认目标房间的尺寸允许全机器人在内活动，并确保底盘停靠点到杯子可达，同时抓取轨迹不使任何机器人几何体越界。

## 最新无全臂语义反例的复现结果

报告 [62](62_unconstrained_semantic_counterexample.md) 与原始实验报告 [31](31_whole_arm_semantic_gcs.md) 记录了协议和数据。2026-10-11 独立运行 5 个条件 × 3 种方法，共 15 次 GCS 规划：

- 4 个正常条件中，Base-only GCS 全部求得连续路径；最小碰撞间距约 1.505 cm，但全臂语义 residual 为 −19.06 cm 或 −29.06 cm，语义不通过。
- 同一条基线路径经 postcheck 检查后全部拒绝，说明事后检查不会自动修复。
- Semantic GCS 的 4 条轨迹均通过离散碰撞和全臂语义审核，semantic residual 约 +2.83 至 +9.74 cm。
- 限制收臂幅度的预设负例在当前姿态族/凸覆盖图中无可达路径；这不是整个机器人 C-space 不可达的证明。
- 这轮重跑了规划和轨迹数值审核，没有重跑 SAP 动力学。

该实验“无语义”仅指移除**全臂房间边界语义**；底盘模式、碰撞几何、起终点和求解器仍保留。它证明这个语义条件会改变可选区域和 GCS 路径，但不是教授 green-room bridge 的完整答案。

## Green-room 原型当前记录的数值

配置和实现位于 `configs/scene_green_room.yaml`、`src/whole_robot_room.py`、`src/green_room_gcs.py`；一次完整规划/参考轨迹结果保存在被 Git 忽略的本地目录 `outputs/results/green_room_v1/`。记录显示：

- 初始构型不在 Room 2 内，whole-robot residual 为 −3.42 m；导航停靠点整机位于 Room 2，residual 为 +0.78 m（房间边界余量含配置的 2 cm margin）。
- 规划由 3 次 GCS solve 组成：收臂底盘导航、固定底盘 7D reach、固定底盘 7D lift；reach 和 lift 分别使用有序 guide-seeded 凸盒区域。不是一个联合的一次求解，也没有从语义谓词自动推断遗漏的动作模式。
- 输出 550 帧可执行参考轨迹；1,655 个插值状态审核通过，导航阶段手臂固定、阶段切换完整构型误差约 `4.6e-22`，操作阶段底盘误差约 `2.8e-17 m`。
- 操作阶段最小 whole-robot 房间 residual 为 +0.710 m，杯子单独检查的最小房间 residual 为 +0.678 m；参考路径的最小碰撞距离约 2.58 mm。这个数值虽为正，但余量较小，不能把“通过碰撞检查”等同于满足 2 cm 墙面安全间隙。
- reach 凸区域独立留出抽检 920 个样本中发现 2 个碰撞构型；lift 的 200 个样本未发现碰撞。抽检覆盖不等于区域内每一点都已认证。
- `green_room_v1` 的结果明确写着 `dynamics_executed: false`。本次 green-room 结果是规划参考轨迹和离散审核，不是新的 SAP 物理执行结果；抓取接触几何审核也不代表接触动力学抓取成功。

## 教授的期望与当前进度对照

| 教授期望 | 当前状态 | 差距 |
|---|---|---|
| 用具体 3D 场景说明机器人任务 | 已有多个 Drake 双房间场景 | 需选定与 green-room 语义完全一致的那个配置。 |
| `whole-robot-inside(green room)` 作为导航目标 | 已有 oracle 检查底盘、手臂、夹爪所有碰撞几何；导航 dock 用 whole-robot residual 验证为正。 | 原型已实现；需进一步确认 room 几何/余量符合教授设想，并保留清晰的场景图和配置。 |
| `whole-motion-inside(green room)` 作为操作期间不变量 | reach/lift GCS 凸盒按运动学上界筛选；参考路径对每个线段作稠密插值审核。 | 已有数值原型；尚非严格连续时间证明，且凸区域留出样本检出了碰撞。需要收紧区域或改进安全验证。 |
| 从房间外初始状态到语义合法区域的连续 motion bridge | 收臂底盘 GCS 从初始构型到 whole-robot-in-room dock，完整构型与 arm 固定姿态接到操作阶段。 | 已以分阶段连接实现；不是自动发现缺失语义/动作的通用 bridge 方法，需解释其区域图、端点和假设。 |
| 说明 Drake GCS 如何使用语义约束求解连续路径 | 有 staged 原型：导航 GCS + 房间内 7D reach GCS + 7D lift GCS；整机语义上界收紧操作凸盒。 | 需要结合本任务画清楚各 GCS 的顶点、区域、顺序边、共享端点和语义审核，并明确这不是一次联合 GCS。 |
| 回答是否找到连续 trajectory | `green_room_v1` 有 550 帧轨迹，1,655 个插值状态的独立审核通过。 | 目前只能说找到通过离散审核的连续参考轨迹；尚未得到本轮 SAP 物理执行证据，且覆盖区域抽检仍有碰撞反例。 |
| 自动从自然语言 command 映射到 predicates | 尚未完成 | 当前仍由人工把任务写成谓词和动作模式；属于后续研究，不是这个 toy example 的首要验收。 |
| 改进 GCS 计算算法 | 使用 Clarabel、IRIS-NP2 和 GCS-Bridging 论文方法 | 尚无新的通用 GCS 算法贡献；教授说高效 GCS 计算可以先作为黑盒，优先理解 C-space。 |

## 下一步应按这个问题定义开展

1. **审查冻结场景和配置。** `scene_green_room.yaml` 已固定一个可用场景；整理房间、门洞、margin、dock、杯子位置和机器人几何示意，确认整个机器人进门和操作范围均符合 toy example。
2. **补全对照。** 在完全相同场景、起终点、参考姿态与碰撞模型下，运行 `--no-semantic` 基线并审计整机房间谓词，和当前 semantic GCS 配对比较。明确不改变核心规划器，只消融 whole-room 语义约束。
3. **收紧碰撞区域。** 对 reach 区域 920 个独立留出样本中的 2 个碰撞反例定位具体 region，修改区域筛选/扩张条件后用新冻结版本重跑；同时达到预设的墙面/障碍安全余量，避免仅凭 2.58 mm 正距离宣称稳健安全。
4. **物理执行。** 运行 SAP 自由杯接触仿真，记录双指接触力、杯子抬升/保持、整机与杯子 room residual、非预期接触和执行器限幅；保持语义审核与一般 delivery checker 的指标区分。
5. **解释 GCS 与 bridge。** 画出导航、reach、lift 三张图或一个明确的分阶段总图，标出端点共享关系、语义激活区间、操作凸盒的 whole-room bound，说明 bridge 是当前由给定导航区域/路径生成的，不是自动语义发现。
6. **固定验收和汇报。** 保留协议、seed、配置快照、完整轨迹、审计与失败结果；回答谓词定义、C-space/GCS 落地、bridge 连续性以及物理执行情况。不要把图不连通直接说成真实 C-space 不可达。

## 结论边界

当前工作已经给出有用的算法和仿真基础，复现了“几何上可行但语义不合意”的 GCS 轨迹，展示了稀疏凸区域图的连通性修复，也建立了教授 green-room 任务的分阶段 GCS 原型。接下来重点是补齐配对消融、区域安全收紧和 SAP 动力学执行，再用清楚的图解释 GCS 与 bridge 如何连接。

在专项验证完成前，不应声称：

- green-room 原型的离散轨迹审核等于连续时间或整区域的严格安全证明；
- 门口“底盘在外、手臂伸入”的任务就是 green-room toy example；
- 当前系统已自动从自然语言推导完整语义计划；
- 有限样本无碰撞就是区域或连续时间的严格安全证明；
- 已证明 GCS 对一般问题具有更高成功率或优于强 baseline。

## 相关材料

- [61 项目阶段总览](61_project_work_summary.md) 与 [LaTeX/PDF 图文总结](61_project_work_summary.pdf)
- [57 门口场景与教授反馈整理](57_professor_feedback_scenario_semantics.md)
- [58 稀疏语义连通性诊断](58_sparse_semantic_refinement_connectivity.md)
- [59 GCS-Bridging 复现](59_gcs_bridging_paper_reproduction.md)
- [60 完整门口 GCS 与 SAP 验证](60_unified_semantic_gcs_bridging_physical_validation.md)
- [62 无全臂语义反例复现](62_unconstrained_semantic_counterexample.md)
- green-room 实现与测试：`src/green_room_gcs.py`、`src/whole_robot_room.py`、`tests/test_whole_robot_room.py`；可复现运行：`.venv/bin/python -m src.green_room_gcs --output outputs/results/green_room_v3`。
- 其他实验入口：`src/arm_boundary_gcs.py`、`src/experiment_gcs_bridging.py`、`src/animate_unified_doorway_gcs.py`
