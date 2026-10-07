# 报告 56：抓放阶段的 mode-specific IRIS 区域与统一 GCS

> 对应执行计划 `reports/55_execution_plan_gap_closure.md` 阶段 1（板书第三层 `valid_whole_body_interaction` 的推进）。
> 代码：`src/manipulation_gcs.py`（核心）、`src/smoke_manipulation_gcs.py`（冒烟）、`src/animate_delivery_demo.py`（集成）、`tests/test_manipulation_gcs.py`（测试）、`configs/scene.yaml` 的 `manipulation_gcs` 节。

## 动机

此前抓放的七个操作阶段（arm_to_pregrasp、approach_cup、close_gripper、lift_cup、lower_into_bin、release_cup、retreat_arm）全部由 IK 插值或笛卡尔路径执行，**不经过任何凸区域**；只有运输段（carry）在 IRIS 覆盖 + GCS 里。教授板书第三层要求交互谓词进 C-space 建模——本报告把操作阶段并入统一的 10 维凸集 + GCS 体系：自由空间模式使用 IRIS-NP2，接触前的 pregrasp 使用有序接触走廊凸集。

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

grasp / release / retreat：参考路径上贪心选种（权重与 carrying cover 相同）→ 每种子生长 IRIS-NP2 区域（复用 `_grow_cover_region`，含独立留出审计）→ 相交区域连边 → 断连时按参考路径补种（上限 8 次）。

pregrasp 接近杯子的末端存在预期指尖接触。若对整个手指 body 做碰撞豁免，IRIS 区域会允许夹爪侧面穿杯的凸捷径；若完全取消豁免，Drake 的接触邻域 IRIS 分离问题又不稳定。因此采用由原参考轨迹相邻点轴对齐凸包构成的**有序 10 维接触走廊**，只保留前向相邻边。它不是整区域安全证书，最终 GCS 轨迹必须通过原尺寸杯子的无过滤主场景审计。

四个模式均由 Drake `GcsTrajectoryOptimization` 求解，使用相同代价度量和速度界；求解后再经过模式检查器和主场景无过滤审计。失败时仍保留显式失败记录，不把回退伪装成 GCS 成功。

## 最终完整场景结果（名义场景，auto_000 候选）

| 模式 | 区域数 | 区域留出审计 | 轨迹审计 | GCS 求解 | 臂关节行程 |
|---|---|---|---|---|---|
| pregrasp | 39 个有序走廊凸集 | 不主张整区域证书 | 0/46 | <0.01s | 3.244 rad |
| grasp | 3（含 1 桥接） | 0/600 | 0/16 | 0.02s | 1.441 rad |
| release | 2 | 0/400 | 0/12 | <0.01s | 1.035 rad |
| retreat | 2 | 0/400 | 0/12 | <0.01s | 1.043 rad |

（文件 `outputs/results/manipulation_gcs_smoke.json`；服务器负载 ~280 下的墙钟时间共约 20s/候选。）

## 测试

`tests/test_manipulation_gcs.py` 的单元与集成检查通过：
- 模式表完整性（附着/夹爪/杯子位形/站位逐项断言）
- 种子选择性质（端点保留、阈值下界、种子必须是参考样本、常数路径拒绝）
- 域构造行为（种子包含、关节限位裁剪、lane 盒与 yaw 界排除）
- 禁用开关不产生副作用
- 操作层可使用独立于 carrying cover 的构型空间裕量
- 简化模式检查器允许、但主场景无过滤审计拒绝的捷径必须回退
- pregrasp 接触走廊参数和有序连边受到回归约束
- grasp 模式端到端集成（真实场景 + IK + IRIS + GCS + 端点精确 + 审计零碰撞）

## 端到端验证

本机 Drake 1.57 最终完整场景结果保存在
`outputs/results/report_ready_gcs_v3/`：

- 首个自动候选 `auto_000` 通过，共 1256 帧；
- 全程碰撞检查通过，最小有符号间隙 2.46 mm；
- 新增独立墙体安全门，要求杯子和机器人对所有墙体均至少保持 2 cm：
  实测携带杯子最小间隙 6.80 cm，整机最小间隙 4.19 cm，分别检查
  110 个附着帧和全部 1256 个机器人帧；低于门槛会拒绝候选；
- 杯子落入 drop-off bin，语义顺序与独立 semantic monitor 均通过；
- 门洞 IRIS 留出审计 0/1000，携带 cover 留出审计 0/3000；
- pregrasp 以 39 个有序接触走廊凸集通过 GCS，轨迹审计 0/46，主场景无过滤审计通过；
- grasp / release 分别以 4 个 IRIS-NP2 mode-specific 区域通过 GCS、模式审计和主场景无过滤审计；
- retreat 先沿杯子—夹爪分离方向退开 8 cm，再上抬 18 cm；以 29 个有序凸集通过直接 Drake GCS，轨迹审计 0/43；
- 四个操作模式均未触发回退，第三层已经闭环到同一次完整 Drake 动画执行。

同时修复了一处配置矛盾：放置/撤退端点按任务定义保留 3 mm
净空，而旧配置继承 carrying cover 的 5 mm IRIS 裕量，导致合法端点在
Drake 1.57 中必然被判为没有内点。操作层现在独立使用 2 mm 裕量；
carrying cover 仍保持 5 mm。

## 自由杯整机接触动力学验证

运动学规划通过后，`src/execute_delivery_dynamics.py` 重新构建离散 Drake
MultibodyPlant。杯子是 0.2 kg 自由刚体，只在初始化时设置位姿；此后没有
attachment 或位姿重置。iiwa 与 WSG 使用受执行器力限约束的 PD 控制，底盘为
施加广义力的理想全驱动 PlanarJoint，接触由 SAP 推进。

第一次 1 ms 执行虽然完成抬升、运输和落箱，但释放撤回时杯子与夹爪掌部出现
3 次非预期接触（首次 3.36 N），因此按原标准判失败。没有把这类接触改成允许项，
而是重新设计 release-retreat GCS 参考：先远离杯子再上抬，并用有序凸走廊防止
GCS 取捷径。修复后的预注册验收结果：

| 条件 | 结果 | 携带相对平移变化 | 非预期接触 | 双侧接触比例 |
|---|---:|---:|---:|---:|
| 1 ms SAP、夹持开启 | 通过 | 5.87 mm | 0 | 100% |
| 0.5 ms SAP、夹持开启 | 通过 | 6.77 mm | 0 | 100% |
| 1 ms SAP、夹持关闭负对照 | 按预期失败 | 4.25 m | 不适用 | 0% |

两种步长的滑移指标相差 0.90 mm，小于预设的 2 mm 复验容差；两次正例均完成
双侧接触抬升、携杯穿门、释放和箱内静止，所有执行器力限满足。完整摘要与网页在
`outputs/results/contact_dynamics_validation/`，原始时间序列分别在
`report_ready_gcs_v3_dynamics*` 三个目录。

## 诚实边界

- grasp / release / retreat 的 IRIS 区域保证是概率式（每区域 ε=0.02，总 δ=0.05 联合上界）；pregrasp 接触走廊不主张该保证。所有轨迹门都是有限采样审计，不是连续时间证书。
- 模式检查器的豁免过滤意味着 IRIS 区域内部允许「夹爪-杯」与「载荷-支撑面」接触；这些接触由逐帧主场景验证兜底。
- pregrasp 接触走廊只对最终 GCS 轨迹提供有限采样安全门，不对走廊内每个点给出概率安全保证；这是与其余三个 IRIS 模式的保证差异。
- GCS 规划模型仍用 attachment 表达 carrying mode；新增执行验证则全程使用自由杯接触动力学。通过只支持当前质量、摩擦、控制器和时间缩放下的模型级结论，不是一般 force-closure 证明。
- 底盘仍是理想全驱动 PlanarJoint，没有轮地接触、执行器时延、传感噪声或硬件验证。
