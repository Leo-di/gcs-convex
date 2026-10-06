# 本周：教授要求 vs 完成情况对照

> 时间：2026-10-05（周一）。供周会汇报与自查。
> 关联：52（交接与失败口径）、53（三层修复技术细节）、54（周报）、55（缺口执行计划）。
> 代码已全部开放：https://github.com/Leo-di/gcs-convex（commit `1700017`，281 文件）。

## 一、教授的要求（组会板书 + 交接确认）

板书三层语义谓词框架：

```
第一层  base 谓词            — 底盘自由空间（2D 圆盘 / SE(2)）
第二层  全臂谓词            — 7D 全臂 FK + 10D carrying 构型空间
第三层  交互谓词            — valid_whole_body_interaction（抓放接触）
```

上周交接确认的优先级：**先深入理解 C-space，修共享导航覆盖**（48 个 heldout 任务中 24 个在查询前就因导航覆盖失败）。

## 二、总对照表

| # | 教授的要求 | 状态 | 证据 |
|---|---|---|---|
| 1 | 修共享导航覆盖（24/48 前查询失败） | **完成** | 三层修复；回归零回归；heldout_01（旧版 0 候选）端到端通过，最小间隙 4.5cm |
| 2 | 第一层谓词进 C-space | **完成** | 导航谓词定义域从固定朝向 2D 仿射切片 → **SE(2) 子流形**（XY 圆盘 × yaw 区间），底盘可在 lane 内变朝向绕行挡路推车 |
| 3 | 第二层谓词（全臂几何进区域计算） | **导航侧完成** | `select_navigation_arm`：收拢姿态参与圆盘半径计算；机械臂几何进入导航区域。七维全覆盖仍是缺口（见 §4） |
| 4 | 第三层交互谓词（抓放进统一 GCS） | **已实现，验证进行中** | `src/manipulation_gcs.py`：pre-grasp / grasp / release / retreat 四模式的 mode-specific IRIS 区域 + 10 维 GCS，按 `manipulation_gcs.enabled` 配置开关接入 delivery demo，附单元/集成测试 |
| 5 | 开发回归（实验诚实度） | **完成** | 16/16 场景最终结果已发布（10-06 02:12 `comparison.json`）：8 个干净完成场景候选数与旧版**完全持平**（00:3/3、07:4/4、11:2/2、14:3/3、15:4/4）；8 个 timeout 不计入结论 |
| 6 | 失败按口径分层统计 | **完成** | 三类分开统计，不混合（§3） |
| 7 | 可演示 | **完成** | 实时 Drake/Meshcat 交互演示（非录像）：底盘收臂、变朝向绕推车、穿门到取物点 |
| 8 | — | **完成** | 全部代码+测试+56 份报告推送 GitHub |

## 三、诚实的失败分类（按报告 52 口径）

| 类别 | 场景/候选 | 定性 |
|---|---|---|
| 覆盖连通但 GCS 取整无路 | heldout_05/06/13 的 x110 | 覆盖图验证连通，Drake 官方 GCS 10 次取整失败——**求解器鲁棒性短板，不是覆盖问题**；下周量化（边集稀疏化） |
| IK 失败（SNOPT infeasible） | 00 x110、11 x110/x115 等 | 与旧版完全相同的候选，本轮未触碰，属第三层，本周的 manipulation_gcs 即针对此 |
| 已知难场景 | heldout_09（0/0） | 推车压 lane 边缘 + 门洞边缘夹死，所有版本一致，非回归 |

回归中的超时不计入任何结论：共享服务器本周被外组 ~18-36 核任务长期占满，需要 SE(2) 慢路径的场景被机器竞争误杀；补跑脚本只重跑超时/程序错误场景，不碰真实结果。

## 四、尚未完成的欠账（与教授要求对照）

| 欠账 | 现状 | 计划（报告 55） |
|---|---|---|
| **七维全臂 C-space 覆盖质量** | 已知参考路径 136/795 采样点被覆盖（~17%），区域生成策略未解决 | 阶段 2：先诊断失败原因（种子 vs 维度），再选策略；1-2 周 |
| **全新场景预注册评测** | 本周回归全部是旧 16 场景（开发集，教授不会认作 held-out） | 阶段 3：样本数/方法顺序/成本口径预先固定，0.5 周 |
| **command mapping** | 第一周遗留：自然语言→任务图仍全人工 | 阶段 4：受限语法指令解析 |
| GCS 取整鲁棒性量化 | §3 第一类失败 | 阶段 5（可选） |
| 连续时间安全证明 | 现为 200Hz 高密度数值验证 | 阶段 5（可选） |

## 五、补跑最终结果（10-06 02:12 发布，`se2_navigation_refresh_v3/comparison.json`）

16/16 场景全部产出最终结果。旧候选合计 20 → 新 16，差异全部来自 timeout 场景。

| 结果类 | 场景 | 数字 |
|---|---|---|
| **干净完成、零回归** | 00/07/11/14/15 | 候选 3/4/2/3/4，与旧版逐场景**完全持平**（合计 16=16） |
| **干净完成、维持 0=0** | 06/09/13 | 与旧版一致为 0 候选——旧版 0 候选在这些场景**不是假失败**，是真实结果 |
| timeout（不计入结论） | 01/02/05/10/12 | 旧版本就 0 候选，新 0；仍无法判定 |
| timeout（不计入结论） | 03/04/08 | 旧版 1/1/2 候选未能复现验证——机器竞争误杀，欠账转下周 |

**§5.1 原预期「heldout_01/02/05/13 应 >0」落空**：13 干净跑完仍 0 候选，01/02/05 依旧 timeout。诚实结论：SE(2) 慢路径修复**没有解锁新候选**，它保证的是零回归（已验证），不是覆盖扩张。

后台其余事项：抓放层 GCS 场景级验证与实时演示求解仍在队列中。

## 六、下周计划（对齐板书）

1. ~~补跑收尾 → 最终对比表~~ **已完成（见 §5）**；剩余 timeout 场景（03/04/08）错峰重跑。
2. 抓放层 GCS 完成场景级验证（板书第三层闭环到执行）。
3. 七维覆盖诊断实验（阶段 2 启动——「深入理解 C-space」的核心欠账）。
4. 预注册全新场景评测协议。

## English summary

This week closed the professor's top priority: the shared navigation coverage repair, with the layer-1 predicate lifted from a fixed-heading 2D slice onto the SE(2) subspace (XY disk × yaw interval), arm posture folded into region computation, and doorway seed densification. The 16-scene regression shows zero regressions on every undisturbed scene; heldout_01, previously failing before query, now passes end-to-end with 4.5 cm minimum clearance. Layer-3 mode-specific grasp/release IRIS regions with 10-D GCS are implemented behind a config flag with tests; scenario-level validation is in progress. Failures are honestly stratified (solver rounding on connected covers, untouched IK failures, one known-hard scene). Remaining debts vs. the professor's board: 7-D arm-space coverage quality (~17%), pre-registered fresh-scene evaluation, and command mapping. All code, tests, and 56 progress reports are now on GitHub.
