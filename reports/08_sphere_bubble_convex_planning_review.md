# 机械臂“小球”、安全区域与凸优化：第一轮文献调研

日期：2026-09-29。范围：核对原论文、作者项目页，并联系当前 10D IRIS/GCS 实现；本轮没有修改规划算法，也没有复现论文性能。

结论：确实存在高度相关的研究，但“小球”至少有三种含义。老师具体指哪篇尚不能确认。最值得优先读的是 **SPARROWS（RSS 2024）** 和 **Safe Bubble Cover（2024 预印本）**：前者解释机械臂的球形运动占据包络，后者解释用安全球构造自由空间走廊。

## 1. 必须区分的空间

| 表示 | 球代表什么 | 所在空间 | 是否直接得到 GCS 可用的安全区域 |
|---|---|---|---|
| 机器人碰撞球 | 包住连杆、夹爪或负载的实体体积 | 物理三维空间 | 否；它表示被机器人占据的空间 |
| 运动占据球 | 包住一段运动可能扫过的体积 | 物理三维空间 | 否；主要用于约束或验证整段运动 |
| 自由空间安全球 | 球内的位置/配置均满足安全条件 | 可能是二维、三维或高维配置空间 | 可以作为凸集，但必须证明球确实安全 |

我们的配置是 `(x, y, yaw, q1, ..., q7)`，因此完整规划区域是 **10 维**。网页上能画出的三维球，不一定就是优化变量所在空间的球。

每个球是凸集，但多个相交球的并集通常不是凸集。先选球序列、再做凸轨迹优化，与联合选择区域的 GCS 不是同一个求解过程；GCS 的凸松弛也不能无条件称作原问题的精确凸解。

## 2. 优先阅读的论文

### A. SPARROWS：球覆盖机械臂的运动占据体积

**Safe Planning for Articulated Robots Using Reachability-based Obstacle Avoidance With Spheres**，RSS 2024。

- [正式会议页面](https://www.roboticsproceedings.org/rss20/p035.html)
- [论文全文](https://arxiv.org/html/2402.08857v1)
- [作者项目与视频](https://roahmlab.github.io/sparrows/)

用重叠球构造 Spherical Forward Occupancy，对机器人运动扫过的体积作保守外包络，再施加避障约束。建议看 Figure 1、Section IV-B 和 IV-D。轨迹参数优化仍是非线性优化，并没有因为使用球就成为 GCS。其安全结论依赖运动、几何和控制等假设成立。

对本项目的启发：应约束执行中可能占据的体积，而不仅验证名义轨迹上的离散姿态。不能只复制球形外观，就宣称继承了论文的连续时间安全保证。

### B. Safe Bubble Cover：用安全球建立走廊

**Safe Bubble Cover for Motion Planning on Distance Fields**，2024 预印本；arXiv 注明提交 ISRR，不能据此认定录用。

- [论文信息](https://arxiv.org/abs/2408.13377)
- [全文及示意图](https://arxiv.org/html/2408.13377v1)

利用距离场和 Lipschitz 性构造安全球，建立相交图，选定走廊后优化 Bézier 轨迹。建议看 Figure 1、Theorem 4.1、轨迹优化部分和 Section 6。论文展示二维、三维导航场景；不是已经验证了我们这种 10D 携物机械臂。

对本项目的启发：球/椭球可以成为区域生成的对照方案。但工作空间距离不能不经转换直接当作关节空间安全半径。

### C. cuRobo 与 GPMP：球近似机器人几何

- [cuRobo 技术报告，Section 3、Figure 3](https://curobo.org/reports/curobo_report.pdf)
- [Continuous-time Gaussian process motion planning via probabilistic inference，IJRR 2018](https://doi.org/10.1177/0278364918790369)

两者都涉及机器人球形近似与环境距离查询。cuRobo 用正运动学计算各球中心并构造碰撞代价；GPMP 系列也用球与距离场构造障碍物项。这是“机械臂上排列许多小球”的典型来源，但不等于形成无碰撞凸配置区域，优化通常仍有非凸性。

对本项目的启发：球可以加速几何查询。若用于安全证明，必须保守覆盖实际碰撞体，不能仅凭视觉贴合或表面采样认定无遗漏。

### D. Neural Configuration-Space Barriers：机械臂配置空间的安全球

**Neural Configuration-Space Barriers for Manipulation Planning and Control**，IEEE TASE 2026。

- [作者项目](https://existentialrobotics.org/bubble_cdf_safe_manipulation/)
- [作者提供的录用稿](https://natanaso.github.io/ref/Long_NCDF_TASE26.pdf)

学习配置空间距离，结合球形区域规划及鲁棒安全控制，并有机械臂实验。这里的球位于配置空间。论文也明确讨论学习误差可能高估半径；神经距离预测本身不是安全证书。

对本项目的启发：它连接了安全球与机械臂，但我们目前有已知 CAD 场景，先做解析/保守基线更容易诊断，不必马上增加网络训练和泛化问题。

### E. Probabilistic Bubble Roadmap：与描述很接近的另一候选

- [2025 年 v1：Probabilistic Bubble Roadmap](https://arxiv.org/html/2502.16205v1)
- [当前 v3：A neural signed configuration distance function for path planning of picking manipulators](https://arxiv.org/abs/2502.16205v3)

v1 Figure 1 同时画出工作空间球形障碍和配置空间安全球，很容易让人留下“很多 3D 小球”的印象。方法以学习距离形成球形节点，建立相交路网，随后优化路径。最新版本聚焦 picking manipulators，不能直接外推为一般 10D 整机验证。引用时应写清版本与标题；学习预测仍需误差控制。

### F. C-IRIS 与 IRIS-NP2：最直接的区域构造参照

- [Finding and Optimizing Certified, Collision-Free Regions in Configuration Space for Robot Manipulators](https://arxiv.org/abs/2205.03690)
- [Faster Algorithms for Growing Collision-Free Convex Polytopes in Robot Configuration Space](https://arxiv.org/abs/2410.12649)

C-IRIS 用代数证书验证区域安全；其有理配置参数化不能直接等同于原始关节角凸性。IRIS-NP2 更注重高效区域扩张和统计终止。当前项目已使用后者，因此“换成 IRIS-NP2”本身不是新贡献。

## 3. 为什么三维球不自动解决机械臂凸优化

设第 i 个碰撞球中心为 `c_i(q)`，半径为 `r_i`；环境距离函数为 `d`，安全裕量为 `m`。约束写成：

```text
d(c_i(q)) - r_i >= m
```

球的形状是凸的，但正运动学 `c_i(q)` 通常非线性，约束对应的关节可行集合通常非凸。末端进入一个安全球，也不能保证肘部、底盘和杯子安全。

以下是针对本项目的推导方案，尚未实现或证明具体机器人的界：若在候选域中可以保证

```text
|h_i(q) - h_i(q0)| <= L_i ||W(q-q0)||₂
```

其中 `h_i` 是扣除球半径后的障碍物净空，则选择

```text
rho <= min_i (h_i(q0)-m) / L_i
```

可以得到该域内的保守安全球/椭球 `||W(q-q0)||₂ <= rho`。需要同时处理：

1. 米与弧度的尺度矩阵 W，以及角度坐标的有效范围。
2. 整个候选域上的运动学变化上界；不能拿种子点雅可比当全域证明。
3. 全部连杆、夹爪、负载和自碰撞对；相对运动需要相应变化界。
4. 与关节限位、语义模式允许域取交集；先保证种子有正裕量。
5. 跟踪误差和杯子平移/转动变化的占据膨胀；一次实验的最大误差不是普遍误差界。

这种构造可能很保守，导致球过小、数量过多或窄门不连通。它首先应是可解释的安全基线，而不是预先承诺优于 IRIS。

## 4. 对当前项目的判断与下一步

本地 `src/iris_region_generator.py` 输出 HPolyhedron，Hyperellipsoid 用于初始区域；当前主体不是一组工作空间小球。此前报告中的区域碰撞抽样 `2/2800` 和门口 `1/1000` 仍待解决。统计碰撞体积分数保证不等于区域内任一点都安全。

[整机执行报告](07_full_contact_execution.md)记录了约 17.5 mm 底盘跟踪误差与毫米级杯子相对位移变化。它们不能直接与最小名义碰撞距离相减，但说明下一步应该把执行偏差通过几何映射纳入规划。

建议实施顺序：

1. **建立保守几何球覆盖与可视化**：连杆、夹爪、不同负载模式分别处理；保留原始几何查询作为独立审计。不能靠删掉碰撞对取得通过。
2. **实现一个加权配置空间安全球/椭球基线**：先固定底盘做 7D 诊断，再扩展到当前 10D；用解析变化界构造半径，明确证书的假设。
3. **在相同任务上比较现有 IRIS-NP2 与球区域 GCS**：统一起终点、碰撞模型、语义模式、时间预算。测总建图时间、碰撞查询数、区域数量、连通率、求解成功率、轨迹质量和独立碰撞审计。零碰撞样本仍不等于证明。
4. **再研究误差包络与混合区域**：借鉴 SPARROWS 的占据外包络思想，将执行误差映射到规划裕量；若球太保守，再讨论多面体/椭球混合。必须单独评估新增算法的收益。

可能的研究问题是：**如何在不同抓取模式下，用可验证的几何与执行误差界，构造足够大且连接良好的凸配置区域，并降低 GCS 规划的总成本？** 这是待验证的研究假设，尚不能称为已确定的新颖贡献。
