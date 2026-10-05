# 三维 Drake 语义 GCS 任务：历史与下次会话交接

## 文档用途

这份文档用于下次继续 `/home/hudi/projects/convex` 的三维 Drake 语义运动规划任务。它按研究问题和证据演进整理既有工作，不替代各轮原始报告、实验协议、日志或结果 JSON。复现实验前，应先读对应报告和其输出目录里的 `protocol.json`；已有结果目录不可复用覆盖。

需要与 `/home/hudi/projects/new-plan` 区分：`convex` 是两房间、移动底盘、iiwa 七轴机械臂、WSG 夹爪和杯子的三维 Drake 场景；`new-plan` 是另一条围绕语义 GCS、2R 和证书的研究线。本交接只覆盖 `convex`。

## 项目是什么

任务是在两间房之间完成“去 Room 2 取杯，再搬到 Room 1 放入箱子”。规划状态视问题而异：空载底盘通常是 `(x,y,yaw)`；搬运阶段可用 `(x,y,yaw,q1…q7)` 的 10D 配置，杯子通过固定抓取变换附着在夹爪上。抓取、放置动作通常由经过几何审核的有限 IK 原语提供。

主要技术链路：

1. 手工结构化场景谓词与任务状态/模式。
2. Drake `GcsTrajectoryOptimization` 或 `GraphOfConvexSets` 选择凸区域路线并优化区域内连续变量。
3. Drake IRIS-NP2 在配置空间生成凸区域；对搬运区域，把附着杯子的几何纳入规划碰撞模型。
4. 独立检查抓取/放置 guard、任务顺序、碰撞、区域成员关系和连续轨迹采样。
5. 部分冻结方案再用自由杯子和接触动力学执行。

这不是一个已证明全局最优、完备或连续安全的通用 TAMP 系统。必须始终区分：任务图可达、求解器返回、采样审核通过、实际接触执行通过，它们是不同级别的证据。

## 演进时间线

| 阶段 | 做了什么 | 结论与转折 |
|---|---|---|
| 场景和语义原型 | [README](../README.md)、[报告 01](01_semantic_modeling_literature_review.md) 建立两房间场景、基础任务序列与 GCS/IRIS 研究框架。初始路径和动作中有不少手工部分。 | Product 图、语义标签、GCS 都有直接相关先例；区域标签必须代表整个区域，而不能只凭中心点或名字。 |
| 抓取可信度修复 | [报告 05](05_simulation_method_review.md) 发现原默认抓取把杯子与夹爪几何过滤，出现厘米级穿透；[报告 06](06_grasp_model_repair.md) 固定抓取姿态和 WSG 开度，增加未过滤几何审计、bilateral fingertip contact 检查和独立语义监测。 | 旧的“抓取成功”动画不能算物理抓取证据。局部自由杯子抬升实验有张开夹爪负对照，但此时完整流程仍主要是运动学回放。 |
| 完整接触仿真 | [报告 07](07_full_contact_execution.md) 在自由杯子、受限 iiwa/夹爪控制和理想全驱动平面底盘下，完成抬升、搬运、释放、落稳；1 ms、0.5 ms 结果一致，张开夹爪对照失败。 | 这是整机接触执行证据，但参考路径放慢三倍；不是轮式底盘、硬件或材料扰动鲁棒性验证。 |
| 圆盘/模式相关凸集探索 | [报告 09–16](09_planar_bubble_gcs.md) 从二维点机器人圆盘原型，推进到空手/携物模式、原场景固定姿态 XY 切片、完整任务拼接及有限姿态圆盘 GCS。 | 圆盘是特定切片/模式下的集合，不等于完整 10D 机械臂 C-space 球。有限姿态圆盘 GCS 在配对比较中区域多、求解更慢、路径更长；保留原 10D GCS 为主要基线。 |
| 语义进入 GCS | [报告 17](17_semantics_inside_gcs.md) 把任务自动机和模式顶点放入同一次 GCS；[报告 18](18_semantic_grasp_candidate_gcs.md) 加入有限抓取候选分支。 | 已能演示语义约束改变合法路线/候选，但抓放原语仍预生成，联合 GCS 的 rounding 不是全局最优保证。 |
| 强基线与任务机制 | [报告 19–29](19_semantic_baseline_long_experiment.md) 扩展候选、检查点、携物禁行和完整任务比较，逐步加入在线 IRIS 与公平预算。 | 早期 9/10 开发集成功不代表方法优越；冻结 40 任务中原方法 5/40，在线修复 11/40，主要瓶颈是原语生成与覆盖。后续机制对照显示任务语义确实影响任务有效性，但强非 GCS 搜索也能满足任务，且通常更快。 |
| 全臂语义与 C-space | [报告 30–32](30_whole_arm_boundary_semantics.md) 从底盘/末端谓词推进到全臂边界、收臂模式和两个姿态分支；[报告 34](34_cspace_chapter3_and_professor_priority.md) 记录导师优先级：先弄懂 configuration space，GCS 算法创新是可选项。 | 两关节切片与七关节 IRIS-NP2/GCS 实验表明非线性全臂谓词可影响路线，但覆盖是有限的；七关节实验里已知较短可行参考路线上一轮覆盖率低，区域覆盖/连接成为限制。 |
| 覆盖、在线接入和完整任务 | [报告 35–42](35_official_iris_gcs_joint_slice.md) 使用 Drake 官方 IRIS-NP2、GCS、VCC 和在线端点区域扩展；[报告 43–46](43_full_task_development_baselines.md) 完成完整任务开发基线、40 任务评估、失败诊断与统一在线修复。 | 在线 IRIS 能修复部分真实端点覆盖缺口，但失败还包含“无有效 IK 原语”和区域图断连；有限区域图失败不等于物理任务不可行。 |
| GCS 角色与强对照 | [报告 47–49](47_semantics_gcs_mechanism_comparison.md) 比较 Product-GCS、同区域 Dijkstra+SOCP、忽略检查点消融、耦合放置目标和统一多目标 Portal。 | 语义约束对正确完成任务有作用；但现有简单任务中，强非 GCS 基线成功率相同且快得多。GCS 的小幅代价收益受有限舍入影响，并存在显著较差的单例；增加 GCS 舍入预算也不保证稳定改进。 |
| 新物理场景与筛查 | [报告 50](50_new_physical_conditions.md) 在 8 个新物理条件、24 个相关任务上比较 GCS10、GCS50 和统一 Portal；[报告 51](51_common_region_screening_regression.md) 修正放杯生成器，并对共享 10D 区域做有限内部/边界筛查及有界 IRIS 重建。 | 报告 50 的成败受到共享区域漏检碰撞影响。报告 51 的已知场景筛查回归中三种查询器均 24/24 通过，筛查库预选接触子集各 8/8 通过；GCS 没有成功率优势，Portal 查询更快。筛查使用旧反例，不是独立迁移测试，更不是区域安全证书。 |

## 报告 51 之后的已有结果

下列结果已存在于输出目录，但还没有正式编号的报告；不要把“工作区有数据”误写为新的公开报告或新算法结果。

### 新随机场景筛查评测

`outputs/results/heldout_transfer_v1/` 包含 16 个预先固定的新随机组合场景、48 个相关任务、288 次规划运行。该批没有向筛查器传入旧碰撞反例；场景仍是同一双房间模板，动作库及共享准备仍受 GCS/IRIS 管线影响，因此不是任意环境泛化，也不是完全独立的无 GCS 冷启动对照。

- 原始库与筛查库下，GCS10、GCS50、Portal 都是 21/48 规划审核通过。
- 筛查库每种方法预选 16 个接触执行机会，各 7/16 通过；实际执行的 21 条全部通过，不能因此写成端到端 100% 成功。
- 每种方法有 24 个任务在共享准备阶段没有候选，另有 3 个参考几何/语义审计失败。全部保留在 48 的分母。
- GCS 查询中位时间约 17 秒，Portal 约 6.9 秒。没有成功率优势；配对差异区间跨零。
- 独立事后诊断把 25 个共享准备拒绝分支中的 25 个首个碰撞都在重建场景中复现，且位于旧导航凸盒内。固定姿态当前几何圆盘只恢复了少数诊断分支，说明共享 bootstrap 导航覆盖是下一瓶颈。

结果和核验入口：[评测摘要](../outputs/results/heldout_transfer_v1/report.txt)、[最终核验](../outputs/results/heldout_transfer_v1/final_verification.json)、[场景设计](../outputs/results/heldout_transfer_v1/design.json)。

### 同场景局部导航加密回归

`outputs/results/refined_transfer_v1/` 是对同一批已诊断 16 场景的建库修复回归，不是独立留出集。局部加密导航后，有效候选数从 20 增至 25、可用放置组合从 119 增至 149；三种方法规划均从先前的 21/48 提至 24/48，预选接触执行各从 7/16 提至 8/16。共享候选/导航建库改善带来的变化不能归因于 GCS 查询器。该结果说明导航覆盖修复有效，但还需要另一批新场景验证迁移性。

详见 [回归报告文本](../outputs/results/refined_transfer_v1/report.txt)、[机制分析](../outputs/results/refined_transfer_v1/mechanism_analysis.txt) 和 [汇总数据](../outputs/results/refined_transfer_v1/analysis.json)。

### 限时携物模块对照

`outputs/results/fresh_timed_carry_v1/` 是另一条更窄的 query-only 携物运动学对照：16 个新随机组合，有限共享动作/区域库，GCS50 对比 Portal50 路线搜索；速度上限、离散路线预算和预先指定时长窗口共同限定问题。不含完整抓放接触执行，也没有加速度/力矩或连续碰撞证书。

预注册主窗口 `relative_115` 下，GCS 9/16、Portal 8/16，但只有一个 GCS-only 分歧，双侧精确 McNemar `p=1`。事后延长 Portal 路线搜索后，原先 GCS 独有成功案例在第 603 条区域路线找到有效路径；这说明结果依赖有限搜索策略，不支持稳健 GCS 成功率优势。不能把其它时间窗口当成独立确认性结果。

详见 [协议](../outputs/results/fresh_timed_carry_v1/protocol.json)、[分析](../outputs/results/fresh_timed_carry_v1/analysis.json)、[失败归因](../outputs/results/fresh_timed_carry_v1/failure_analysis.json) 和 [完整摘要](../outputs/results/fresh_timed_carry_v1/report.txt)。

## 当前实现地图

- 场景和配置：`src/build_scene.py`、`src/scene_spec.py`、`configs/scene.yaml`。
- 任务语义/监测：`src/semantic_predicates.py`、`src/task_semantics.py`、`src/semantic_monitor.py`。
- 空载底盘 GCS：`src/gcs_base_planner.py`。
- 携物 10D GCS：`src/gcs_whole_body_planner.py`、`src/iris_region_generator.py`、`src/carry_mode_gcs.py`。
- Product/候选联合 GCS：`src/semantic_product_planner.py`、`src/semantic_gcs_experiment.py`、`src/semantic_candidate_gcs.py`。
- 强非 GCS Portal + SOCP 对照：`src/full_task_portal_socp.py`、`src/global_portal_socp.py`。
- 新物理条件评测：`src/prepare_transfer_scenes.py`、`src/transfer_scene_experiment.py`、`src/screen_carry_regions.py`。
- 留出集与局部覆盖修复：`src/heldout_transfer_pipeline.py`、`src/refined_transfer_experiment.py`；限时携物对照：`src/fresh_timed_carry.py`、`src/timed_carry_bezier.py`。
- 真实接触执行：`src/execute_delivery_dynamics.py`；全任务主入口：`src/animate_delivery_demo.py`。

各项实验都有冻结源码/输入快照或哈希；复现时优先使用实验目录中的 `protocol.json` 和 `source_snapshot/`，不要直接用当前 `src/` 代表历史运行源码。

## 已证实与未证实

可以说：

- 这个三维 Drake 场景已经有官方 IRIS-NP2 区域生成、任务/模式图、空载和携物 GCS、有限抓取/放置原语、独立语义监测以及完整自由杯子接触执行流程。
- 语义状态会改变允许动作与可接受路径；忽略有要求的检查点可以产生任务违规路线。
- 在线 IRIS 或局部建库修复能修复部分覆盖/连接失败；收益依赖共享动作库、障碍布局和端点覆盖。
- 抓放接触执行在若干预选名义条件下通过，支持的是当前摩擦、几何、控制器和理想平面底盘模型下的结果。

不能说：

- GCS 普遍比强 Dijkstra/Portal + SOCP 基线更快或更容易成功。
- GCS rounding 返回全局最优；报告中的下界/上界有有限松弛、舍入和数值容差。
- IRIS-NP2 抽样审核通过就意味着整个区域安全；已知存在区域内部碰撞反例。
- 场景或有限区域图无路就证明真实任务不可行；IK、动作库、bootstrap 和区域覆盖失败必须分别统计。
- 21 条或 24 条已执行回放通过等于总体端到端 100% 成功率；未执行的预选机会要留在分母中。
- 规划 attachment 模型与自由杯接触执行模型完全一致，或当前底盘代表轮式移动机器人/真实硬件。

## 下次会话建议

**首要任务：修复共享 bootstrap 的当前几何导航覆盖，再做新的冻结评测。** 最新 16 场景留出批里，24/48 任务在查询前就因共享准备无候选而失败，另有 6/48 个导航/语义图断连；局部加密回归已显示候选覆盖可提高结果，但它仍是同场景修复证据。把这一层修好，才能让查询器比较落在更多真实可规划任务上。

建议顺序：

1. 先读 `outputs/results/heldout_transfer_v1/design.json`、`outputs/results/heldout_transfer_v1/final_verification.json`、`outputs/results/refined_transfer_v1/mechanism_analysis.txt`，确认各次失败发生在哪个 bootstrap 阶段。
2. 用共同方案修当前几何导航覆盖，保留语义 lane 和碰撞几何；不允许按查询方法生成不同的便利输入。把无候选、图断连和轨迹审核拒绝分开记。
3. 在旧 16 场景做开发回归；不要称作 held-out。确认改动不破坏原来的已通过任务和接触回放。
4. 另行预注册一批新的物理组合场景和任务/接触子集；固定样本数、方法顺序、GCS rounding、Portal 路线预算、区域筛查和总成本口径。不要把上一轮反例作为测试特征送入新方法。
5. 继续报告共享准备成本、筛查成本、查询时间、共同成功路径代价、完整执行数和所有失败原因。按物理场景聚类统计；查询任务不能当独立场景。

若研究问题转为证明 GCS 的独特价值，应设计真实的连续耦合因素，并给强 Portal 基线相同的路线搜索预算。仅把速度上限改成时间窗并不足以保证出现差异：限时实验唯一的 GCS-only 成功被更深的 Portal 搜索找回。新的对照要先验证其差别来自联合建模，而不是有限搜索上限。

## 常用检查

在 `/home/hudi/projects/convex` 下：

```bash
LCM_DEFAULT_URL=memq:// .venv/bin/pytest -q
```

实验脚本通常要求输出目录尚不存在；正式评测耗时较长，不要在交接时自动重跑。开始修改前记录工作树状态；修改后先运行受影响测试，再考虑全套回归。
