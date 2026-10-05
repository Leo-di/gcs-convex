> 后续修复：本文件保留修复前的审查记录。当前默认抓取及新增局部接触验证见 [06_grasp_model_repair.md](06_grasp_model_repair.md)。

# 当前方法与相关研究的对照审查

日期：2026-09-29。审查对象是当前工作区的规划、建模及验证流程，不只是动画观感。

## 判断

作为研究起步阶段的语义规划原型，Drake + 自动机 + IRIS/GCS + 运动学 attachment 是可以解释的技术路线；但当前模型有实质性缺口，不能仅凭场景看起来完整、测试通过或优化器返回成功，就报告“机器人可靠抓取并安全执行”。

本次发现杯子与被过滤的夹爪几何有厘米级穿透。这使过去的“成功案例”必须限定为简化模型下的规划/回放结果，而不是物理抓取成功。此前报告中的 1/3、2/3 完成数同样受此限制。应先修复抓取几何，再扩大候选搜索或性能统计。

## 与相关研究是否相同

| 参考 | 相同或相近的思想 | 当前实现与该工作不能等同之处 |
|---|---|---|
| Kurtz & Lin，T-RO 2023，[LTL-GCS](https://arxiv.org/pdf/2301.07773) | 自动机、区域标签、GCS 连续规划 | 当前先选离散路线再做各段 GCS；不是直接复现其联合构造，也不能继承其定理。该论文 III-B 要求标签对整区域成立 |
| Garrett et al.，ICAPS 2020，[PDDLStream](https://ojs.aaai.org/index.php/ICAPS/article/view/6739) | 符号决策需要连续几何求解支持 | 当前只是固定任务序列下的有限站位重试，没有实现该系统的通用动作语言、stream 关系和规划算法 |
| Graesdal et al.，RSS 2024，[Contact-Rich GCS](https://www.roboticsproceedings.org/rss20/p132.html) | 模式序列与连续运动组合 | 该工作研究接触模式及准静态动力学，包含平面推动；当前杯子 attachment 没有这种接触物理模型，不应把两者当等价实验 |
| MIT，[Basic Pick and Place](https://manipulation.csail.mit.edu/pick.html) | 理想化抓取后物体相对夹爪固定，是可用的初级建模假设 | 教材从物体与夹爪相对位姿构造动作；当前仅位置约束的 IK 加任意朝向、过小开度，不能靠 attachment 把错误抓取变成正确抓取 |
| Drake，[接触仿真示例](https://drake.mit.edu/tutorials/hydroelastic_contact_basics.html) | 使用相同建模平台 | 官方示例使用 Simulator 推进系统；本项目主要 SetPositions/SetFreeBodyPose/ForcedPublish，是回放而非闭环动力学执行 |

这些工作解决的问题、系统假设和证据不同。并非所有规划论文都必须做接触动力学或硬件，但选择运动学抽象时，至少应保证抽象内的几何一致性，并明确结论只覆盖该抽象。

## 发现一：夹爪—杯子碰撞过滤掩盖了大穿透（最高优先级）

`build_scene.py` 将自由杯子与全部夹爪几何排除碰撞；attached-cup 场景又将附着杯子与整台机器人几何排除。过滤预期的接触可以有明确用途，但过滤整个模型并不能证明被排除的几何关系有效。

对默认原场景 trace 做独立查询，使用 `ComputeSignedDistancePairClosestPoints`。官方说明该接口不遵守碰撞过滤，因此可以专门审计被排除的几何对。[Drake QueryObject 文档](https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_query_object.html)

实测结果：

- 433 帧中最严重的杯子—夹爪几何距离约 **-0.0476115 m**，发生于第 274 帧，右手指 collision box。
- 夹爪闭合末帧（190）杯子与夹爪本体距离约 **-0.0182542 m**。
- 首个超出审计容差的穿透出现在第 148 帧，即接近动作期间，而不只是闭合后的预期接触。
- 审计发现 2410 个违规“帧—几何对”，不等于 2410 个违规帧。

数值针对当前圆柱杯子和夹爪碰撞几何，不是视觉网格或真实硬件的精确测量。即便如此，几厘米的穿透不能解释成接触数值容差。

相关原因：原 `plan_pick` 只约束抓取点位置，不约束夹爪朝向；闭合关节位置是 `[-0.006,0.006]`，与半径 0.05 m 的杯子不匹配。若只看中心点对齐或后续 attachment 一致性，会漏掉这种错误。

## 发现二：运动学展示被称为仿真时容易夸大证据

`animate_delivery_demo.py` 设置配置和杯子位姿，调用 ForcedPublish 显示。没有用反馈控制器驱动机器人、没有动力学积分、没有接触力与摩擦验证。杯子“跟随夹爪”和“释放后保持不动”来自脚本规则。

因此可以说“运动学场景、规划轨迹、离散验证”，不能据此说力闭合、不会滑落、会稳定落底或真实机器人可以跟踪。

放置支撑当前仍是 8 mm 高度容差，不表示真实支撑反力存在。动画统一使用帧索引乘 0.05 s，而不是完整保留每段原优化时间。此次默认回放的离散最大关节速度约 1.07 rad/s，没有观察到超过配置的 1.2 rad/s；但这不证明加速度、力矩或闭环跟踪可行。

## 发现三：碰撞/语义验证的保证弱于名称暗示

IRIS-NP2 的官方概率定义约束区域内碰撞体积比例，并非每一点都安全。[Drake IRIS 文档](https://drake.mit.edu/doxygen_cxx/group__planning__iris.html)

本项目原场景 cover 审计仍有 3/2800 碰撞样本。即使 Bézier 控制点限制在这些区域内，也不能据此证明整条真实机器人轨迹连续无碰撞。

轨迹检查是采样帧上的几何查询，并受过滤规则限制；task monitor 根据记录的运动学状态验证自定义规则，也不是自动生成并验证任意 LTLf 规格。碰撞区域的证据、轨迹证据和任务证据必须分别汇报。

## 发现四：现有搜索与对照仍较小

候选生成依赖人工区域和启发式参数；默认只改变 XY，不搜索底盘 yaw 或杯体抓取方向。每次动作顺序仍固定为取—搬—放，无法体现复杂语义规划的优势。

三个场景、每种方法一次运行只适合诊断。固定基线和自动方法的候选预算不同；没有跨随机种子、多障碍布局或任务分支的统计对照，也没有复现 LTL-GCS/PDDLStream 作为外部基线。不能把目前端到端集成视为新 GCS 算法。

## 本次已实施的纠正

1. 新增 `src/grasp_audit.py`，单独审计所有杯子—夹爪几何对；只对命名的指尖球允许 0.5 mm 数值容差，手掌和手指盒不允许穿透。
2. 主流程每帧记录该审计，结果单列 `grasp_geometry_satisfied` 和 `physical_grasp_validated`，后者始终为 false。旧 `collision_free` 字段只保留原过滤模型的含义。
3. `validation.require_grasp_geometry=true` 时，错误抓取会在 IRIS 之前被拒绝，后续路径若出现穿透也拒绝。默认 false 只用于保留可观看的旧原型，并明确报告其失败审计，绝非新的安全设置。
4. 新增可配置的抓取朝向约束及对应语义 residual；IK 失败结构化区分 pregrasp、grasp、lift、carry、preplace、place。
5. 新增可离线打开的 `review_scene.html`，包含明确的运动学与抓取审计提示。实时 Meshcat 继续保留供观察，不能用动画外观代替上述数值结论。

## 抓取几何修正的诊断结果

`src/diagnose_grasp_ik.py` 对原场景和杯子右移 5 cm 场景，各取 6 个自动站位，比较 legacy 和一个明确的侧抓姿态，共 24 个端点诊断。

侧抓试验使用世界 RPY=(0,0,0)、抓取位置容差 0.2 mm、朝向容差 0.005 rad、开度关节位置 ±0.055 m、闭合位置 ±0.0475 m。这些是实验设置，未作为默认配置替换旧演示。

原场景部分站位的抓取端点穿透从厘米级降至约 **0.17 mm**，仅发生于允许小容差的指尖几何，端点审计可通过。但后续 carry/place IK 仍失败；右移场景的这组侧抓设置也未成功。没有任何侧抓方案被宣称为完整取放成功。

这些结果说明：限制合理朝向和开度能改善抓取端点几何，却也暴露了后续可达性/碰撞约束的耦合。失败原因已经细分，但尚未证明是哪条几何约束使全部方案严格无解。不能通过放宽碰撞过滤、增大支撑容差或继续使用任意 attachment 来掩盖它。

## 推荐后续优先级

1. 固定有意义的物体—夹爪抓取变换与接触面，联合考虑抓取、携带和放置姿态候选；严格审计需成为研究实验的接受条件。
2. 将过滤缩小到指定的接触几何对，并按模式启用；验证物体与手掌、非接触手指和机器人其他部分的关系。
3. 完成严格几何模式下的原场景及扰动场景，然后再扩大对照。若目标仅为运动学 TAMP，可先不做动力学，但必须明示。
4. 若要声称真实执行，再接入控制器、Simulator、接触/摩擦参数及释放后的稳定性验证。用真实时间参数执行轨迹。

## 产物与运行

- 在线原型：本次启动的 `http://127.0.0.1:7000/`，通过已有 SSH 转发访问。
- 离线网页及原型审计：`outputs/results/live_review/review_scene.html`、`grasp_geometry_audit.json`。
- 新流程完整运行：`outputs/results/audited_pipeline/`。
- 24 个端点诊断：`outputs/results/grasp_ik_diagnostic/summary.json`，每种设置有配置快照。

```bash
python -m src.grasp_audit outputs/results/live_review/delivery_trace.json \
  --config outputs/results/live_review/scene_config_snapshot.json \
  --output outputs/results/live_review/grasp_geometry_audit.json
python -m src.diagnose_grasp_ik --output-dir outputs/results/grasp_ik_diagnostic_new
```

第一条命令预期返回非零，因为旧抓取未通过独立审计。这是正确暴露失败，而不是工具安装异常。
