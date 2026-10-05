# 在线 IRIS 接入完整抓放（2026-09-30）

在线端点扩展已从固定底盘七维子实验接入完整任务的 **10D 携物模式**。本次重新求解语义 GCS，并将返回路线送入自由杯子的接触动力学执行；并非重放此前路线。

## 协议与比较范围

复用 `semantic_candidates/library/x110` 的已检查取放原语、14 个携物区域和原场景空手圆盘库。固定候选 x110，不进行候选优劣比较。为验证实际触发在线扩展，明确留出包含抬杯后起点、放置前终点的区域 0 和 10，保留其余 12 个区域。输入、端点、留出索引及哈希在运行前保存。

这是**受控端点区域留出实验**，不是自然发生的失败修复统计，也不是新场景泛化测试。离线库本身曾由完整参考路径生成；它的成本和先验不应隐藏。未采用新七维随机查询的区域，未把杯子留在桌上进行空手规划。

无在线扩展时，同一个语义 GCS 图没有到接受状态的路径，两个携物端点均未覆盖。启用在线扩展后：

| 项目 | 本次结果 |
|---|---:|
| 实际 IRIS-NP2 调用 | 2 |
| 接受的新区域 | 2 |
| 在线接入及区域审核 | 6.82 s |
| GCS 求解 | 2.31 s |
| 返回路径使用新区域 | 12、13（零起始索引） |
| 完整参考帧数 | 459 |
| 加密几何审核 | 4,581 点，0 违规 |
| 最小环境间距 | 2.49 mm |
| 完整语义 / 抓取几何 | 均通过 |

## 算法与审核

状态为 `(x,y,yaw,q1,…,q7)`，夹爪闭合状态固定。根据实际抓取原语的 `X_GC` 创建携杯碰撞场景。端点未覆盖时最多调用一次正式 Drake `IrisNp2`；求解器 Ipopt，禁止非线性求解失败时直接添加超平面。起始球半径沿用 0.006，关节局部域半径 0.48 rad，保留原语义空间域和关节限位。具体参数随每个调用保存。

粒子 2,000，epsilon 0.0025，单次 delta 0.005，最多 6 次迭代；每个区域由携物碰撞检查器独立抽样 1,000 点，再在原始自由杯子模型中检查种子与另 1,000 点，恢复不受规划模型过滤影响的杯子—夹爪审计。两区所有上述检查均通过。采样接受不是整个区域安全证明；此前七维查询 53 的反例仍然成立，不能因为本轮通过而撤销该限制。

在线模块拒绝已知抽样碰撞或不含携物几何的缓存。新区域只属于当前查询；区域拒绝不能进入 GCS，不对失败端点无限重试。若端点仍未覆盖或区域图不连通，停止并保留失败记录；本轮没有增加桥接搜索或用手工路线绕过 GCS。

后续沿用正式 Drake `GraphOfConvexSets`、Clarabel 松弛与 restriction、10 条取整候选，取放动作、空手路径和携物路径进入同一个任务图。携物顶点优化十维入/出配置，凸集内线段及跨区域连续性由约束处理。动作原语仍是预先生成的 IK 运动，空手圆盘仍固定机械臂姿态；不是每个任务模式均使用在线 IRIS。

## 接触执行

新路线重新运行 `execute_delivery_dynamics`，1 ms 步长、10 ms 控制周期，71.70 s 模拟时间。杯子是自由刚体，初始化后位姿重置次数为零，无 attachment；底盘仍是理想全驱动平面模型。

- 双侧接触抬杯、携物回大房间、释放后箱内落稳均通过。
- 携物记录的双侧接触比例 100%；相对夹爪最大平移变化 3.33 mm。
- 超过 0.1 N 的非预期接触记录为零。
- 最大底盘位置跟踪误差 16.94 mm，最大机械臂关节误差 0.01211 rad。

这是一次名义条件运行，不能用于估计完整任务成功率、连续时间安全或硬件可靠性。规划时序经现有统一重定时送入控制器，不是含完整动力学约束的轨迹优化。

## 复现与网页

```bash
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m src.run_online_iris_delivery --output outputs/results/online_iris_delivery_new
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m src.execute_delivery_dynamics outputs/results/online_iris_delivery_new/reference --output-dir outputs/results/online_iris_delivery_new/dynamics --html
.venv/bin/python -m src.publish_online_iris_delivery --source outputs/results/online_iris_delivery_new
LCM_DEFAULT_URL=memq:// .venv/bin/pytest -q tests/test_online_carry_iris.py tests/test_online_iris_attachment.py tests/test_semantic_candidate_gcs.py
```

输出目录必须不存在，避免覆盖失败或成功的旧证据。`--keep-endpoint-regions` 可关闭受控留出，在已覆盖时应避免多余的在线调用。

本次数据：`outputs/results/online_iris_delivery_v1/`。网页：<http://localhost:7001/online_iris_delivery/>，默认展示这次新路线的接触执行，可切换规划参考并跳到各取放阶段。相关测试 9/9 通过。浏览器验证了两类记录、8 次阶段跳转及播放/暂停，无脚本错误；截图和检查结果随实验保存。

下一步是在场景和抓取关系保持兼容的前提下改变实际查询，检验自然出现的覆盖缺口；改变障碍物或抓取变换时必须重新审核缓存，不能沿用旧区域的安全结论。更广的成功率比较应在这些条件明确后开展。
