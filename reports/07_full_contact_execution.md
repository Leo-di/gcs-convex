# 从 attachment 回放到整机接触执行

日期：2026-09-29。承接 [抓取修复报告](06_grasp_model_repair.md)。

本轮增加了独立的动力学执行模式：真实积分 iiwa、WSG 和一个理想全驱动平面底盘，杯子始终是自由刚体。使用修正后的规划参考，完成闭合抓取、抬升、跨房间搬运、落箱、张开夹爪和撤回。**这不是给运动学回放换标签：运行中不设置机器人位置、不重置杯子姿态，也不焊接杯子。**

## 模型与控制

- `build_scene(..., dynamics_time_step=...)` 单独建立离散动力学场景；规划场景仍保留原用途。
- 动力学模式恢复杯子与所有指尖、掌部及取物桌的接触，禁止传入 attached-cup 参数。
- iiwa 与 WSG 使用 Drake SAP 内置 PD，重力前馈通过关节执行器输入，与反馈共同受力矩/力上限约束。杯子不接受补偿力。
- iiwa 增益 `Kp=3000, Kd=100`；执行器力矩上限来自模型，依次为 `176,176,110,110,110,40,40 N·m`。
- WSG 增益 `Kp=5000, Kd=20`，每指力上限 `20 N`。闭合控制参考为 `±0.040 m`，产生有限预紧；实际指尖位置由接触决定。参考位置不是强制穿透位置。
- 底盘通过 3-DOF PlanarJoint 受广义力驱动；平移 PD `5000/500`，偏航 `3000/200`，各平移轴限幅 `500 N`，偏航 `200 N·m`。这不是轮式机器人轮地接触或非完整约束模型。底盘—地板过滤对应平面支撑假设。
- SAP 不支持多自由度关节的关节限位约束，因此动力学模式去掉 PlanarJoint 人工位置限位，同时单独检查实际底盘状态是否仍在原工作区界限内。规划限位保持。
- 参考每帧原间隔 `0.05 s`，执行放慢三倍，前置 `1 s` 稳定、末尾 `2 s` 保持，总模拟 `67.8 s`。控制周期 `10 ms`；状态记录周期 `50 ms`；接触力检查周期 `10 ms`。阶段仍按参考时间调度，尚非接触事件驱动的状态切换。

## 两个不能忽略的修复

**IK 分支跳变。** 撤回路径此前逐点求解后，最后直接拼入另一个预放置 IK 解。两者末端位姿接近，但关节相差最高约 `0.708 rad`。这在动画中不显眼，动力学跟踪误差却达到约 `0.270 rad`。改为在同一初值延续路径上求解最后的目标位姿，不再强行切换关节分支。修复后的整机执行最大关节跟踪误差约 `0.0118 rad`。新增回归检查防止此类末端跳变。

**落箱接触抖动。** 纯 point-contact 模式中，圆柱杯底在箱底出现持续数值抖动；1 ms 与 0.5 ms 下都未通过原先 `0.01 m/s` 落稳阈值。没有放宽阈值，而是为杯子及支撑几何添加 hydroelastic 表示：杯子 compliant，resolution hint `0.01 m`、modulus `1e7 Pa`；桌子、箱子、地板 rigid。其它接触对保留 point-contact fallback。

该选择依据 [Drake Hydroelastic Contact User Guide](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html)：hydroelastic 表示接触面及压力分布，需同时启用接触模型与几何属性；只有打开模型开关而不设置几何属性，不能得到所期望的面接触。这里的刚度和摩擦仍是仿真假设，未经真实材料标定。

## 接受条件

所有条件必须同时满足：

1. 观察到杯子抬升超过 `10 cm`，且双侧手指接触力均大于 `0.05 N`。
2. 观察到杯子进入 Room 1，仍有双侧接触；搬运阶段全部记录点都必须有双侧接触。
3. 最后 `1 s` 的所有记录点满足杯子在箱内、箱底接触力大于 `0.05 N`、两指接触力小于 `0.05 N`、杯子线速度小于 `0.01 m/s`，角速度小于 `0.05 rad/s`。
4. 控制采样点不得出现超过 `0.1 N` 的非预期接触；支撑接触也须符合阶段（例如搬运阶段杯子不能仍留在取物桌上）。
5. 实际底盘状态在工作区界限内，采样到的净执行器输出不超过上限。

这些是有限采样下的执行验收条件，不是连续时间安全证明。不能把未检测到超阈值接触改写成严格无碰撞。

## 实验结果

| 条件 | 抬起并跨房间 | 持续双侧接触 | 释放落稳 | 最大关节误差 | 最大底盘位置误差 | 搬运相对位移变化 |
|---|---|---|---|---:|---:|---:|
| 1 ms，闭合 | 通过 | 100% 记录点 | 通过 | 0.01176 rad | 17.44 mm | 5.69 mm |
| 0.5 ms，闭合 | 通过 | 100% 记录点 | 通过 | 0.01176 rad | 17.52 mm | 6.89 mm |
| 1 ms，张开对照 | 未抬起 | 无夹持 | 未送达 | — | — | 不适用 |

两个正例未检测到超过阈值的非预期接触；所有执行器在限制内。两种步长下任务结论一致，但位移变化不同，**没有证明完全无滑移或数值收敛**。此处全程任务接受条件不要求严格保持物体—夹爪变换，不能与上一轮局部试验台的 `5 mm` 滑移阈值混用。局部实验和本实验的指尖预紧参考也不同。

新参考轨迹的 4,321 点加密运动学审计通过，回归测试 54/54。上述动力学实验另外运行，不能把 54 项单元/回归测试的数量当成物理成功次数。

负例没有偷偷绑定杯子：机器人照参考运动，但杯子留在取物桌，所有任务完成条件失败。程序保存失败结果并返回非零退出码。

## 复现

在仓库根目录：

```bash
LCM_DEFAULT_URL=memq:// .venv/bin/python -m src.animate_delivery_demo --no-wait --output-dir outputs/results/full_dynamics/reference
.venv/bin/python -m src.audit_replay outputs/results/full_dynamics/reference
LCM_DEFAULT_URL=memq:// .venv/bin/python -m src.execute_delivery_dynamics outputs/results/full_dynamics/reference --html --output-dir outputs/results/full_dynamics/validated
.venv/bin/python -m src.execute_delivery_dynamics outputs/results/full_dynamics/reference --time-step .0005 --output-dir outputs/results/full_dynamics/validated_fine
.venv/bin/python -m src.execute_delivery_dynamics outputs/results/full_dynamics/reference --no-grip --output-dir outputs/results/full_dynamics/open_control
```

负对照退出码 1 是预期结果。`--point-only` 可复查点接触支撑的诊断基线。各目录内 `result.json` 保存状态、接触力、实际执行器输出、接受条件及结果；`config.json` 保存场景设置。早期失败诊断保留在 `nominal/`、`preloaded/`、`fine/`，不属于最终验收结果。

已有复查页面可用以下命令加入新视图：

```bash
.venv/bin/python -m src.build_review_page outputs/results/grasp_fix_web --benchmark outputs/results/grasp_fix_benchmark/summary.json --full-dynamics outputs/results/full_dynamics
```

原 `http://localhost:7000/` 现在默认显示“整机接触仿真 3D”，还可切换之前的运动学回放和局部抓取动力学实验。展示的是保存的真实积分记录，不是浏览器在线控制器。

## 下一阶段应解决的问题

本轮是给定场景、给定摩擦与质量、较慢时间参数下的一次完整接触执行验证，不是一般可执行性证明或真实机器人验证。材料参数扰动、物体姿态与尺寸误差、接触事件驱动阶段切换、轮式底盘动力学尚未验证。

更直接的研究问题是把**跟踪误差和物体滑移反馈到规划裕量**：观察到的底盘误差约 `17.5 mm`，而运动学轨迹最小环境距离仅约 `2.48 mm`，二者不是同一个几何量，不能简单相减，也不能据此认为存在鲁棒安全保证。需要通过机器人几何/雅可比把误差映射为占据区域膨胀，再研究 GCS 区域与轨迹的鲁棒约束。

IRIS 门口区域 `1/1000`、名义区域覆盖 `2/2800` 的碰撞抽样问题仍未在本轮解决，不能因动力学演示成功而忽略。
