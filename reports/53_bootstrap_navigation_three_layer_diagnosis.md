# 共享 bootstrap 导航断连的三层诊断与修复

本轮按 [报告 52](52_task_history_and_handoff.md) 的首要任务（修复共享 bootstrap 的当前几何导航覆盖）推进。结果：断连根因分三层，**三层全部修复**并通过单元测试（6/6），最难挡车场景（heldout_01）覆盖图已连通。本轮同时产出了一个展示绕障导航的 3D 回放页面。

## 起点：断连证据链

- `heldout_transfer_v1` 中 24/48 任务因共享准备无候选失败；事后诊断（`posthoc_diagnostic_verification.json`）显示 25 个碰撞分支全部位于旧导航凸盒内，用固定姿态当前几何圆盘重试仅 3/25 成功，其余 22 个覆盖图断连。
- 挡路障碍是随机场景中的移动推车（门宽 1.10–1.20 同时变化）。

## 第一层：产品图把 outbound 锁死在选中的最短路线上

`plan_delivery_candidate` 原先把空载导航的允许区域限制为 `product_plan.regions_in_mode(seeking_pickup)`，即产品图**选中的那条**空间路线。实测 heldout_01：

```text
regions_in_mode(seeking_pickup) = [room1_central_lane, doorway_lane, room2_pickup_lane]
```

不含 `room1_upper_lane`。推车挡住 room1_central_lane 时，outbound 无法上绕。而 carrying 模式恰因放置点在 room1_upper_lane 才返回了全部 4 条 lane——这是选中路线的偶然产物，不是语义差异。

**修复**：outbound 改为放行该场景 `gcs_base_planning.regions` 的全部 lane。语义正确性由独立 monitor 逐帧检查真实谓词保证，不依赖区域预选择。产品图仍决定任务顺序（pick→place），连续规划器在语义合法区域内自主选路绕障——这是 TAMP 分层的正确职责划分。

## 第二层：推车与隔墙之间走廊的覆盖密度

放行全部 lane 后，base 配置（门宽 1.2）+ 推车挡路场景仍断连。逐点诊断发现走廊内圆盘半径只有 0.05–0.07，0.35 的网格间距接不上（相邻圆盘中心距 0.40，半径和仅 0.12）。

**修复**：在门洞接近走廊加对称 0.1 步长加密种子，覆盖**推车可能占据的整个 x 范围（[-1.7,0]）和上下两条绕行带（y∈[-0.35,0.55]）**。第一版只加密了上方走廊（为推车在南侧的场景设计），推车在北侧的场景（如 heldout_03，从下方绕行）仍然断连；对称化后解决。base 配置下完整 GCS 求解成功（最小距离 0.0757，路径 5.29m）。

**16 场景连通性验证**（修复后逐场景检查 SE(2) 覆盖图）：15/16 连通。唯一断连的 heldout_09（推车中心恰在 y≈0.02 压住通道，且 room1_upper_lane 右边缘 x=-0.55 处底盘距推车仅 4cm、圆盘半径低于阈值）在旧版与 refined 版中也都是 0 候选，不是回归，属于已知难场景。对照：旧 heldout 运行有 8 个场景因导航失败得 0 候选；refined 版仍有 6 个 0 候选场景。

## 第三层（机械臂 home 姿态在门洞处撞隔墙）

用**实际 heldout_01 配置**（门宽 1.1046）复现，加密种子后仍断连（doorway 0/34 可达）。逐点距离检查定位到真正的碰撞对：

```text
x=-0.55（room1_upper→room1_central 过渡带），机械臂为 home 姿态：
  y=0.40  距离 +0.084   最近障碍对: shared_wall_upper / iiwa_link_6
  y=0.45  距离 +0.034
  y=0.50  距离 -0.016   ← 撞墙
  y=0.55  距离 -0.066
```

碰撞体是 **`iiwa_link_6` 与 `shared_wall_upper`**：home 姿态手肘（joint4=-1.2）半伸，臂向前探出门洞平面。

排除性验证：

1. **变朝向无效**：推车上方可行 y 窗口在 yaw=0 时为 [0.32,0.305]（空集）；实测 yaw=0.4/0.8/1.57 只会更差（底盘 y 向占宽从 0.45 增至最大 0.6）。yaw=0 已是 y 向最窄姿态。
2. **简单收臂姿态无解**：搜索了 joint2∈{0,0.45,-0.2} × joint4∈{-1.2…-2.05} × joint6∈{0,0.4,0.75} 共 30+ 组合，没有一组同时满足关节限位（joint4 限 ±2.09）、无自碰撞、且在撞墙点距离≥0.02。肘折叠到 -2.5/-3.0 直接超限且自碰撞。

**第三层修复（收臂导航姿态）**：扩大搜索空间（加入 joint1 侧转与 joint2 后倾）后找到有效解。关键发现是 home 姿态的问题不在手肘折叠量，而在**肩关节前倾 + 腕关节上翘把小臂探向门洞平面**；把 joint2 从 +0.45 后倾到 -0.3、joint6 归零即可收回。在 heldout_01 全路径关键点（起点、推车上方、走廊、门洞、room2 接近）上验证：`[0,-0.3,0,-1.2,0,0,0]` 等 37 组姿态全部满足关节限位、无自碰撞、最小环境距离 ≥0.05（home 在同样点上有 -0.016）。

实现为 `select_navigation_arm(scene, base_positions)`：按固定顺序尝试收臂候选，取第一组在给定底盘位置（起点+pickup）距离 ≥0.03 且无自碰撞的姿态，全部失败则回退 home。接入点：

- `plan_geometry_outbound` / `plan_geometry_outbound_se2` 新增 `arm_q` 参数（默认 home，向后兼容）；
- `plan_delivery_candidate` 在 outbound 前选择姿态，用于 outbound 覆盖、navigate_to_pickup 帧和 pregrasp 过渡起点（安全 pregrasp 规划器从该起点做碰撞检查路径）。

**验证**：heldout_01（此前最难场景）SE(2) 覆盖图连通（247 区域、241 可达、四条 lane 全部打通）；单元测试 6/6 通过（含姿态选择与 home 回退测试）。姿态选择是有限确定性搜索，不是姿态证书；采样 outbound 审计与密集回放审计仍是门。

结论与 [报告 34](34_cspace_chapter3_and_professor_priority.md) 记录的导师优先级一致：**空载导航的机械臂姿态是决定性因素**，这正是老师板书第二条 `do_not_cross_boundary(arm, doorway)` 的含义。

## SE(2) 扩展的定位

本轮实现了 `plan_geometry_outbound_se2`：区域 = 当前几何 XY 圆盘（裁剪到 lane box）× yaw 区间，`GcsTrajectoryOptimization(3)` 官方求解，按 yaw 切片探测距离。对推车挡路场景实际求解出的路径 yaw 保持 0（绕行是 y 向的，不需要转向），说明 yaw 自由度对此类场景不是决定性因素，但对其它推车位置仍有价值。实现要点：

- 只在切片 yaw 处探测距离（避免窄门洞圆盘被 ±yaw 探测过度收缩到消失）；
- 相邻切片 yaw 区间互相重叠，支持原地转向；
- 固定朝向版失败时自动回退到 SE(2)；
- 有采样审计兜底，不是连续时间安全证书。

## 3D 回放页面

`record_se2_demo.py` 生成自包含 Meshcat 3D 回放：推车挡路场景（heldout_01 推车位置），SE(2) 规划绕障路径，1001 帧。本地页面 `http://localhost:7001/se2_demo/index.html`。

## 环境问题（如实记录）

本机同期有他人的 GPU 训练任务占满 CPU（load 一度 16）。相同规模的 2D 基线 GCS 求解从诊断记录的约 4 秒劣化到约 297 秒，SE(2)（183 区域）约 419–474 秒。覆盖构建与连通性检查本身都在 1 秒内。本轮据此没有盲跑 16 场景回归（会重复断连并烧掉数小时），已清理中途产生的无效结果目录。

## 产物与复现

- 实现：`src/geometry_outbound_gcs.py`（`plan_geometry_outbound_se2`、走廊加密种子、`select_navigation_arm` 收臂姿态选择、`arm_q` 参数）、`src/animate_delivery_demo.py`（outbound 全 lane 放行 + SE(2) 回退 + 收臂姿态接入导航帧与 pregrasp 起点）、`src/record_se2_demo.py`（3D 回放）。
- 测试：`tests/test_geometry_outbound_gcs.py` 6/6 通过（含新增 `_se2_yaw_slices`、姿态选择与 home 回退测试）。
- 页面：`outputs/results/grasp_fix_web/planar/se2_demo/index.html`。

```bash
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q tests/test_geometry_outbound_gcs.py
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.record_se2_demo --se2 \
  --cart -1.25 -0.165 0.41 --start -3.0 0.0 0.0 --goal 1.2 -0.1 0.0 \
  --output outputs/results/grasp_fix_web/planar/se2_demo
```

## 旧 16 场景开发回归（进行中，中期结果）

回归用 `se2_navigation_refresh_v3`（与 heldout_transfer_v1 相同 16 场景、相同 4 个 pickup 候选、无测试反馈）。**共享服务器同期有其他用户的 ~18 核负载任务，导致大量场景被超时误杀**（机器竞争，非算法失败），已用补跑脚本 `se2_patch_scenes.py`（只重跑 failure=timeout/program_error 的场景，不碰真实结果）分批补全。

无干扰完成的场景，**全部与旧版持平、零回归**：

| 场景 | 旧版候选 | 新版候选 | 备注 |
|---|---|---|---|
| heldout_00 | 3 | **3** | 补跑完成（x115/x120/x125；x110 为 IK 失败，与旧版相同） |
| heldout_07 | 4 | **4** | |
| heldout_11 | 2 | **2** | 失败均为 IK（SNOPT），本层未触碰 |
| heldout_14 | 3 | **3** | |
| heldout_15 | 4 | **4** | |
| heldout_09 | 0 | **0** | 已知难场景（推车压中心+lane 边缘夹死），非回归 |

待补跑场景（01,02,03,04,05,06,08,10,12,13）：全部是推车挡路、需要 SE(2) 慢路径的场景，负载下每候选 45–90 分钟，正在第二轮补跑中。其中 01/02/05/13 旧版 0 候选、覆盖图已验证连通 + heldout_01 端到端已单独验证通过（outbound 最小距离 4.5cm，全链路含 IK/IRIS/carry GCS/采样验证通过），预期补跑后候选数 > 0。

失败分层（按报告 52 口径分开统计，不混合）：
- **导航断连**：heldout_09（推车恰在 y≈0.02，room1_upper 右边缘 x=-0.55 处底盘距推车仅 4cm，圆盘半径低于 0.025 阈值全部被跳过）——所有旧版本也都是 0 候选，属已知难场景。
- **覆盖连通但 GCS 取整无路**：heldout_06/13 的 x110（"Official GCS found no path"）——大图上求解器取整失败，是后续改进点（更稀疏的边集或更多取整尝试），不是覆盖问题。
- **IK 失败（SNOPT infeasible）**：与旧版相同的候选（如 heldout_00 x110、heldout_11 x110/x115），本层未触碰，非回归。

## 下一步

1. 等补跑完成（5 并行、6 小时/场景超时，预计一天内），汇总最终新旧候选数对比并更新本报告。
2. 关键验证点：heldout_01/02/05/13 补跑后候选数应 > 0（旧版 0），即为三层修复的净增益。
3. 之后按报告 52 流程预注册新场景评测；若个别场景仍失败，预期集中在 IK 与 IRIS 种子。
