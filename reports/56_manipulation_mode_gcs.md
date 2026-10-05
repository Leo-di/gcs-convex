# 报告 56：抓放阶段的 mode-specific IRIS 区域与统一 GCS

> 对应执行计划 `reports/55_execution_plan_gap_closure.md` 阶段 1（板书第三层 `valid_whole_body_interaction` 的推进）。
> 代码：`src/manipulation_gcs.py`（核心）、`src/smoke_manipulation_gcs.py`（冒烟）、`src/animate_delivery_demo.py`（集成）、`tests/test_manipulation_gcs.py`（测试）、`configs/scene.yaml` 的 `manipulation_gcs` 节。

## 动机

此前抓放的七个操作阶段（arm_to_pregrasp、approach_cup、close_gripper、lift_cup、lower_into_bin、release_cup、retreat_arm）全部由 IK 插值或笛卡尔路径执行，**不经过任何凸区域**；只有运输段（carry）在 IRIS 覆盖 + GCS 里。教授板书第三层要求交互谓词进 C-space 建模——本报告把操作阶段并入同一套「IRIS-NP2 区域 + GCS」体系。

## 四个操作模式的碰撞模型

每个模式是 10 维约化状态 `q=(x,y,yaw,q₁…q₇)` 上的一组冻结碰撞几何（夹爪开合与杯子位形冻结进 `parameterization`，由 `MODE_SPECS` 声明）：

| 模式 | 杯子 | 夹爪 | 站位 | 覆盖的相位 |
|---|---|---|---|---|
| pregrasp | 自由，桌上（障碍） | 张开 | pickup | arm_to_pregrasp → approach_cup |
| grasp | 附着（X_GC 焊接） | 闭合 | pickup | lift_cup |
| release | 附着 | 闭合 | dropoff | lower_into_bin |
| retreat | 自由，箱内最终位形（障碍） | 张开 | dropoff | retreat_arm |

close_gripper / release_cup 是 2 自由度夹爪插值、机械臂不动，不在 10 维状态内，保持定臂回放（诚实记录，不假装进了 GCS）。

**关键工程点（Drake 碰撞过滤语义）**：
1. `SceneGraphCollisionChecker` 要求按 body 对均匀的过滤，而 `build_scene` 的「杯-指尖」过滤不均匀（指尖只是手指 body 的一部分）→ 模式场景对「自由杯 × 全夹爪」整体豁免。
2. grasp 模式的种子在杯-桌接触处（杯子本来就在桌上）、release 模式在杯-箱近接触处 → 对「附着载荷 × 支撑面（桌/箱）」豁免——这是抓取/放置的预期接触。
3. 这些豁免**只放宽区域检查器**；demo 的逐帧最终验证仍跑在无过滤的主场景几何上，安全性等级不变。模式场景独立构建（缓存复用），绝不弱化共享场景的审计。

## 方法

每个模式：参考路径（原插值/笛卡尔路径）上贪心选种（权重与 carrying cover 相同）→ 每种子生长 IRIS-NP2 区域（复用 `_grow_cover_region`，含独立留出审计）→ 相交区域连边 → 断连时按参考路径补种（上限 8 次）→ 10 维 GCS 求解（与 carry 相同的代价度量、速度界）→ 轨迹逐采样过模式检查器审计。失败时回退参考路径并记录（`fallback_to_reference`），不静默吞错。

## 冒烟结果（名义场景，pickup_x1_1 候选）

| 模式 | 区域数 | 区域留出审计 | 轨迹审计 | GCS 求解 | 臂关节行程 |
|---|---|---|---|---|---|
| pregrasp | 5 | 0/1000 | 0/24 | 0.02s | 2.496 rad |
| grasp | 3（含 1 桥接） | 0/600 | 0/16 | 0.02s | 1.441 rad |
| release | 3（含 1 桥接） | 0/600 | 0/16 | 0.01s | 1.360 rad |
| retreat | 2 | 0/400 | 0/12 | 0.01s | 0.925 rad |

（文件 `outputs/results/manipulation_gcs_smoke.json`；服务器负载 ~280 下的墙钟时间共约 20s/候选。）

## 测试

`tests/test_manipulation_gcs.py` 7/7 通过：
- 模式表完整性（附着/夹爪/杯子位形/站位逐项断言）
- 种子选择性质（端点保留、阈值下界、种子必须是参考样本、常数路径拒绝）
- 域构造行为（种子包含、关节限位裁剪、lane 盒与 yaw 界排除）
- 禁用开关不产生副作用
- grasp 模式端到端集成（真实场景 + IK + IRIS + GCS + 端点精确 + 审计零碰撞）

## 端到端验证

（进行中：`outputs/results/manipulation_gcs_e2e/`，全任务含语义顺序、逐帧无过滤碰撞验证、放置结果。完成后更新本节。）

## 诚实边界

- 区域保证是概率式（每区域 ε=0.02，总 δ=0.05 联合上界），轨迹门是有限采样审计，不是连续时间证书。
- 模式检查器的豁免过滤意味着区域内部允许「夹爪-杯」与「载荷-支撑面」接触；这些接触由逐帧主场景验证兜底。
- 抓取仍是运动学附着，非接触力学。
