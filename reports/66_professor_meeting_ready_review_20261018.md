# 教授当前想看的最小演示：已实现内容与汇报顺序

日期：2026-10-18。依据 `plan/id23_note.txt` 的研究讨论，尤其 24:11–42:40；结果来自报告 65 的最终 IRIS/Drake 实验。此次整理没有改变规划或场景，也没有重新运行实验冒充新结果。

## 1. 当前应该交付什么

教授当前明确要求：解释必要语义约束的定义、实现，以及是否找到连续轨迹。对此，已有两房间任务已具备可展示证据。下一步应先用同一页面完整说明，而不是继续堆叠场景复杂度或追求 GCS 成功率优势。

汇报页面：`http://localhost:7005/professor_review_20261018/`。
它整合任务定义、约束作用范围、motion bridge、IRIS/GCS 接口说明、完整接触动画、实际状态语义曲线、正负例和局限。

## 2. 两个语义约束

令 q=(x,y,yaw,q_arm)，B(q) 为底盘、机械臂、夹爪全部碰撞几何的并集，R2 为房间竖直棱柱，安全余量 δ=0.02 m。

- `whole-robot-inside(Room 2)`：B(q) 在 R2 内且保留 δ；施加于导航终点/操作起点。
- `whole-motion-inside(Room 2)`：操作轨迹的每个时刻都满足该配置谓词。

实现映射：

| 问题 | 实现 |
|---|---|
| 某配置的整机房间谓词如何计算？ | `whole_robot_room.py`：对每个机器人碰撞几何求房间面法向上的支撑值，取最坏 residual |
| 怎样选择目标？ | `semantic_refinement.py`：候选底盘 XY 与抓取方向；整机筛选、IK、路径与执行验证，失败继续尝试 |
| 布尔谓词怎样影响连续规划？ | `green_room_gcs.py`：保守运动学界约束整个关节域的房间余量；不是只给一个 true/false 标签 |
| 碰撞自由凸区域如何得到？ | `green_room_iris.py`：正式 Drake IRIS-NP2，在上述语义域内切割；模式区分空手/持杯 |
| 如何连接区域？ | 检查真实多面体相交，建立 GCS 边；必要时新增 IRIS 起点/共享配置区域；不伪造边 |
| 如何验证物理执行？ | 自由杯 SAP 接触执行，记录真实机器人状态、双指接触和物体运动；实际整机房间几何另行审计 |

## 3. Motion bridge 的准确含义

当前语义有效集合 S 只定义合法操作配置。初始机器人在 Room 1，q0 不属于 S，所以需要一段无碰撞导航把 q0 连到 S 中的合适配置。

导航 bridge → 整机入室停靠配置 → 房间内操作。

这是给定任务阶段下的连接搜索，不是系统从自然语言自动发现“进入房间”动作。导航过程中不要求整机已经在 Room 2；操作阶段才要求持续保持。碰撞约束始终存在。

IRIS 的 `missing_start` 补区是另一个层面的区域覆盖修复，不能与一般语义计划补全混为一谈。

## 4. 可以如实汇报的结果

- 当前任务找到连续参考轨迹，并通过真实接触仿真。
- 3 次官方 GCS：导航、reach、lift。没有宣称一次联合优化。
- 46 + 11 个 IRIS 区域，5700 个区域审计样本无碰撞/抓取几何/房间语义反例。
- 整机几何检查包含 20 个碰撞几何；操作实际最小房间 residual 约 0.710 m。
- 双指抬杯与保持通过，非预期接触记录为 0。
- 物理日志生成的曲线展示导航时 residual 为负是允许的，操作时持续为正。
- 目标合法但门被封住的独立负例被拒绝，说明目标筛选没有被误当作可执行性保证。

实际状态曲线来自 `outputs/results/semantic_iris_physics_20261018/candidate_000/dynamics_trace.json`；使用独立的静态 Drake 场景重算每帧整机几何，不修改原始物理日志。

图件：`outputs/results/professor_review_20261018/actual_semantic_trace.png` 与 SVG；来源和实际阶段时刻见 `evidence.json`。

## 5. 下次会议建议按这个顺序讲

1. 展示任务图：先整机进房间，再在房间内抓杯，纠正门外伸手版本的任务差异。
2. 给出两个谓词，强调目标条件与全过程条件的区别。
3. 说明候选选择、IK、导航连接、IRIS 区域、GCS 的接口，不把它们混称为 GCS 自动完成。
4. 播放完整接触动画，展示实际语义曲线。
5. 说明当前只证明该具体任务可解；请教授对这个问题定义和接口理解提出反馈。

可用英文描述：

> I have implemented the two constraints as whole-robot geometry checks: containment is a goal condition for navigation and an invariant during manipulation. The planner searches docking and grasp candidates, connects the initial configuration to a feasible dock with a navigation motion bridge, and uses IRIS-NP2 and GCS for the room-constrained arm motions. We found a continuous trajectory and validated it in Drake contact simulation. The task sequence is still supplied; general completion of sparse semantic plans remains the next research step.

## 6. 接下来的研究工作，而不是当前演示的必备前提

优先：冻结任务语义与场景，设计 geometry-only、goal-only 和 full-motion-semantic 的公平对照。需要允许前两者在同一候选空间选择，不可把现有仍强制整机入室的 `--no-semantic` 选项直接当成完全无语义基线。

其次：从多阶段给定语义序列出发，研究候选失败、区域覆盖不足、缺少过渡动作之间的区分及补全方法。

旧凸盒 vs IRIS 已有对照验证的是区域质量，不等于语义消融；闭门负例验证连接失败处理，也不等于整个配置空间不可行证明。不可用当前单个成功例声称一般成功率优势或论文贡献已完成。
