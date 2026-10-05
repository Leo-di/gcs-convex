# 原始三维场景中的正式 IRIS-NP2 与 GCS 关节切片实验

本轮按用户要求保留研究算法本体，直接调用 Drake 的 IRIS-NP2 与 GcsTrajectoryOptimization，完成报告 34 建议的同步关节空间展示。实验维度为两个真实关节；没有用圆盘覆盖、栅格搜索或手工折线替代区域生成和规划。

网页：<http://localhost:7001/joint_cspace/>。可以切换两种条件、播放规划参考轨迹、点击关节空间检查同步三维姿态。点击吸附到最近诊断网格点；该网格不参与规划。网页不依赖持续运行的 Meshcat 服务，沿用 7001 静态服务。

## 算法依据与实现范围

|环节|研究来源与本轮实际调用|
|---|---|
|GCS|[Marcucci 等，Motion Planning around Obstacles with Convex Optimization](https://arxiv.org/abs/2205.04422)；Drake `GcsTrajectoryOptimization`，三次 Bézier、凸松弛、取整及固定路径凸优化。|
|区域生成|[Werner 等，Faster Algorithms for Growing Collision-Free Convex Polytopes in Robot Configuration Space](https://arxiv.org/abs/2410.12649)；Drake `IrisNp2`，Ipopt 非线性碰撞反例搜索。|
|与导师第二篇关系|[Petersen 与 Tedrake 的 IRIS-NP](https://arxiv.org/abs/2303.14737)是前序工作。本机 Drake 1.57.0 提供后续 IRIS-NP2，本轮不声称复现原版 IRIS-NP 或论文全部 benchmark。|
|参数化|官方 `IrisParameterizationFunction`，将两个独立关节映射到完整 plant 位置向量，保留自动微分；参见[官方接口说明](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1_iris_parameterization_function.html)。|

自主编写的部分是场景适配、种子提议、语义几何编码、结果审核与界面，不是重新实现一个简化版 IRIS 或 GCS。该实验使用文献方法的正式实现，但不是完整论文复现，也不是新的算法贡献。

固定底盘 `(-0.6,0,0)`，开放 `iiwa_joint_1` 和 `iiwa_joint_2`；其余关节为原 home 的 `[0,-1.2,0,0.75,0]`。范围读取实际 plant 限位。起点 `(-1.6,0.45)`、终点 `(1.6,0.45)` rad。杯子保持原桌面位姿，没有抓取或 attachment。

原模型允许杯子与部分指尖碰撞几何接触，不符合本次无接触模式，也不满足 SceneGraphCollisionChecker 对 body-pair 筛选一致性的要求。本轮适配器重新启用所有杯子—机器人碰撞对，未通过扩大碰撞豁免解决接口问题。此调整同时用于两组方法和审核场景，不修改原项目全局抓取配置。

## 语义怎样进入区域生成

目标是所有机械臂和夹爪碰撞几何均满足 `x≤0`。在独立规划场景副本里，将 `x≥-0.002` 的一侧编码为禁止进入的盒体，范围为 `x∈[-0.002,19.998]`、`y,z∈[-20,20]`。该盒体远超固定底盘机械臂的可达范围，因此在本实验作用范围内表示半空间禁区。固定底盘本身完全处于允许一侧，杯子是环境物体。

这是一种针对空间包含谓词的几何编码，不是把所有一般语义都转成实体障碍物。该禁区仅存在于规划副本；原始三维展示和独立审核使用原场景，调用 `ArmBoundaryOracle` 对每个 arm/gripper 盒体或球体计算精确支持函数 residual。

IRIS 因而在生成区域时就处理该限制，而不是事后给区域中心贴一个语义标签。仍需注意：IRIS-NP2 是概率方法，整个区域满足约束不是本轮严格证明的结论。

## 设置与结果

区域提议依次使用起点、终点及固定随机种子 713 的 400 个均匀样本，只对未覆盖的可行点调用 IRIS；没有参考路径或栅格规划器。每组至多接受 8 个区域、至多尝试 16 次。IRIS 每次配置 1000 particles、epsilon=0.005、delta=0.00125、最多 5 次外循环、30 次分离循环、0.01 rad 构型空间 margin。这些是算法输入参数，不是本轮独立证明的全覆盖安全概率。

每个返回区域另取 500 个样本，使用原始碰撞查询和独立全臂 oracle 审核；违反对应约束的区域不进入 GCS。GCS 使用 Clarabel、最多 20 条取整路径，代价为 Bézier 控制多边形长度上界加 `0.05×duration`，关节速度限幅 ±0.6 rad/s。

|指标|仅避碰|避碰 + 全臂语义|
|---|---:|---:|
|IRIS 尝试 / 接受区域|6 / 5|9 / 8|
|区域生成调用时间合计|15.06 s|24.94 s|
|区域独立审核样本|2500|4000|
|区域样本碰撞数|0|0|
|区域样本语义越界数|343（该组不约束语义）|0|
|GCS 求解时间|26.62 ms|38.41 ms|
|轨迹审核点数|1067|1298|
|轨迹碰撞点数|0|0|
|轨迹语义越界点数|537|0|
|最小全臂语义 residual|−19.39 cm|+4.62 cm|
|采样路径长度|3.2000 rad|3.5816 rad|

生成时间只计 IRIS 调用，不包含种子筛选、独立审核和网页导出；GCS 时间为 SolvePath 调用。单实例时间不作为性能基准结论。路径长度是采样估计，不能与控制多边形上界混为一谈。

两组都保留了一次失败：相同种子 `(-2.77899,1.66661)` 周围不足以容纳指定构型空间 margin，IRIS 抛出异常，完整错误已记录。语义区域生成还出现部分非线性碰撞反例子问题未成功求解的 Drake 警告；本轮不把算法正常返回等同于安全证书。所有返回区域均通过另取样本的审核后才用于规划。

这组对照显示：同样使用正式 GCS，加入全臂语义的区域生成会改变构型空间可用集合和最终运动。它没有隔离或证明 GCS 相对于其他规划器的优势。

## 验证与边界

两项集成测试通过：参数映射仅改变两个关节且保留正确自动微分；在 84 个构型上，语义规划副本的判定与“原碰撞判定且全臂 residual≥2 mm”一致，并包含物理可行但语义不合法的反例。该测试不是所有构型上的形式证明。

浏览器检查覆盖两种轨迹切换、点击网格、播放、Meshcat 时间与所选构型同步；没有 JavaScript 错误。首次截图发现视角被墙遮挡，随后改为从上方查看。显示数据包含 1681 个诊断网格姿态与两组各 241 个轨迹姿态，全部由同一真实模型的正运动学得到。

Bézier 路径施加一阶参数连续性；不同分段时长仍可能导致物理时间速度跳变。没有加入加速度、力矩或接触动力学约束，也未新增动力学执行。因此这是研究算法的运动规划实验，不是动态可执行性证明。后续应在独立关节维度和任务模式上扩展，而不是将当前二维切片重新命名为完整 10 维规划。

实现：`src/joint_cspace_research.py`、`src/report_joint_cspace_research.py`、`src/joint_cspace_research_view.html`。原始结果及设置：`outputs/results/joint_cspace_research_v1/`，包括全部区域 A/b、失败记录、轨迹、浏览器检查和源码快照。

```bash
# 指定新目录重跑，已有实验目录会拒绝覆盖。
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m src.joint_cspace_research --output outputs/results/joint_cspace_research_repeat
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.report_joint_cspace_research
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_joint_cspace_research.py
```
