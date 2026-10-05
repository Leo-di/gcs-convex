# 语义条件与独立监测：第一阶段实现

日期：2026-09-29。承接 `01_semantic_modeling_literature_review.md`。

## 已完成

- 移除 pickup lane 的整区域 near 标签、dropoff lane 的整区域 at_dropoff 标签。动作的 `candidate_regions` 只描述候选站点，实际几何条件另行检查。
- `src/task_semantics.py` 统一取物距离、房间、夹爪闭合、抓取点对齐、箱内足迹及支撑高度余量。杯子按圆柱碰撞几何计算有姿态的范围，箱子为开口容器，允许杯口高于箱沿。
- 动画生成前检查已求解的抓取/放置端点。guard 失败会报错，不再把离散动作候选当作几何可行证明。
- `src/semantic_monitor.py` 根据原始杯子/夹爪位姿、夹爪位置和 attachment 关系重建 pick/release 事件，检查切换前后条件、携带期间位姿关系、释放后杯子静止、最终张开夹爪及撤回。累积违规，后续成功不能覆盖先前失败。
- 保存完整观测、配置快照、首个违规帧与残差；支持离线重放，失败返回非零状态。保留旧的顺序检查作为单独字段。

## 监测发现并修复的真实问题

第一次完整运行在 place guard 失败：考虑杯子实际姿态后的支撑余量约为 -0.001615 m。原流程携带时使用 FK 得到的杯位姿，释放时却改成理想目标位姿，掩盖了 IK 误差。

修复保持支撑容差 8 mm 不变；仅最终放置 IK 的位置/姿态容差从 5 mm / 0.06 rad 收紧到 2 mm / 0.02 rad。释放位姿改为求解后 FK 与抓取变换计算出的实际位姿，不再跳到目标。更严格的 0.5 mm / 0.005 rad 配置曾求解失败，没有采用。

几何支撑仍按容差判定，允许小的高度间隙，不表示杯子在接触动力学中已经稳定落底。后续若需真实支撑，应单独引入接触或动力学验证。

## 验证结果

- 测试 25/25 通过，包含实际 Drake IK 的释放位姿连续性回归测试。
- 完整演示 433 帧，逐帧碰撞与新语义监测通过。
- 独立离线重放 433 帧通过；pick 在第 191 帧、release 在第 343 帧（零起始编号）。
- 本次未过滤几何对的最小距离约 3 mm，不是墙面专项距离。
- IRIS cover 仍为 14 区域，独立审计 3/2800 碰撞；该问题未在本阶段修复，生成时的求解警告仍存在。

对真实记录注入五种观测故障，新监测器全部拒绝：

| 注入故障 | 首个违规帧 | 识别条件 |
|---|---:|---|
| 抓取点改到距离桌子 1.74 m 的角点 | 191 | near_pickup 余量为负 |
| 抬升过程中解除 attachment | 211 | 提前释放，不满足箱内条件 |
| 附着状态下杯子相对夹爪漂移 3 cm | 211 | attachment 位姿不一致 |
| 释放帧杯子横向移动 40 cm | 343 | 箱内足迹失败 |
| 在释放前截断轨迹 | 342 | 未完成释放 |

这些是记录级故障注入，不是物理重仿真或规划器对照实验。前四项特意保留旧缓存谓词，检查新监测不盲信缓存；第五项说明杯子入箱但未释放时，旧顺序检查也可能已经通过。结果在 `outputs/results/semantic_counterexample_audit.json`。

## 复现

```bash
cd /home/hudi/projects/convex
source .venv/bin/activate
python -m pytest -q
LCM_DEFAULT_URL=memq:// python -m src.animate_delivery_demo --no-wait
python -m src.semantic_monitor outputs/results/delivery_trace.json \
  --config outputs/results/scene_config_snapshot.json \
  --output outputs/results/semantic_monitor_replay.json
```

## 明确边界与下一步

本阶段是当前单杯任务的有限状态监测，不是通用 LTLf 编译器，也不是完整连续时间证明。对象身份由单杯场景固定；输入 FK、碰撞余量及 attachment 关系仍来自仿真器，没有独立传感器证据。没有改变 attachment 的无效释放命令无法从状态边沿识别，需要后续记录命令事件。

所有条件仍在离散帧上检查，尚未证明整个凸区域的全部语义标签。监测能发现违规不等于规划器会自动修复；当前没有多候选动作搜索或几何失败后的重规划。所有阶段的 mode-specific IRIS、有限轨迹规格编译、任务分支对照实验留待下一阶段。

建议下一步增加明确的命令事件日志和多个抓取候选，再将几何 guard 失败反馈给候选动作选择；同时推进区域审计，不应先宣称整个 product graph 已获安全证明。
