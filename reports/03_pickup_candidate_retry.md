# 多候选取物站位与失败反馈

日期：2026-09-29。承接第二阶段的几何 guard 与语义监测实现。

## 本阶段变化

取物站位不再固定为 reference_path 的最后一点。`pick_task.candidates` 配置三个底盘候选，按顺序尝试 `(1.0,-0.1,0)`、`(1.1,-0.1,0)`、`(1.2,-0.1,0)`。这里的“抓取候选”具体是不同底盘站位下的 IK 解，尚不是对不同杯体抓取面、抓取方向或物体身份的搜索。

每个候选依次经过：

```text
距离与动作区域检查
→ 去程 GCS
→ 抓取 IK 与后续携带/放置 IK
→ 动作端点 guard
→ attached-cup IRIS 与 whole-body GCS
→ 完整轨迹的碰撞、事件顺序及独立语义监测
→ 全部通过才接受并发布动画
```

任何上述规划/验证阶段的 RuntimeError 或明确的候选拒绝会触发下一候选。TypeError、配置 ValueError 等非预期错误会直接上报，不静默当成几何不可行。失败阶段、错误内容和候选耗时保存到 `candidate_attempts.json`。每次尝试结束后恢复原机器人/杯子位置和场景时间。

主要实现为 `src/candidate_planner.py`；完整候选求解与验证在 `animate_delivery_demo.py::plan_delivery_candidate`。主场景和 attached-cup 场景共享本次配置，支持 `--config` 和 `--output-dir`。

## 实际运行结果

| 候选 | 结果 | 原因/证据 |
|---|---|---|
| x=1.0 | 拒绝 | 抓取 IK 成功，但后续放置 IK 返回失败；阶段 delivery_ik |
| x=1.1 | 拒绝 | 后续放置 IK 返回失败；阶段 delivery_ik |
| x=1.2 | 接受 | 全流程求解、433 帧碰撞与语义监测通过 |

这不是人为在代码中按候选名称制造失败，使用的都是实际 Drake/SNOPT 求解结果。也不应把局部 IK 求解失败解释成该站位严格无解。

本机一次运行：两个失败尝试合计约 0.090 秒，包含 IRIS、GCS 和逐帧验证的候选搜索总计约 20.16 秒。该时间不包括主场景创建与 Meshcat 发布，不能与此前仅最终 GCS 的约 1.56 秒直接比较。

只保留第一个候选的对照运行以非零状态退出，报告 `exhausted`；启用三个候选则选择 `pickup_x1_2` 成功。它说明有限候选重试在这一个实例中有效，不构成跨场景性能结论，也不说明原来固定 x=1.2 的方案失败。

验证：33/33 测试通过。新增测试覆盖最终验证失败后的重试、场景状态恢复、真实 IK 候选耗尽、距离 guard 提前拒绝、非预期错误传播及配置合法性。最终验证失败后的调度测试使用受控回调；本次真实全流程的前两个失败均发生在 IK 阶段，未声称真实复现了 IRIS/GCS 阶段失败后的成功重试。

完整成功轨迹再次通过离线语义重放。IRIS cover 审计仍有 3/2800 碰撞样本，未在本阶段修复。

## 复现与产物

```bash
cd /home/hudi/projects/convex
source .venv/bin/activate
python -m pytest -q
LCM_DEFAULT_URL=memq:// python -m src.animate_delivery_demo --no-wait \
  --output-dir outputs/results/candidate_retry
python -m src.semantic_monitor outputs/results/candidate_retry/delivery_trace.json \
  --config outputs/results/candidate_retry/scene_config_snapshot.json
```

- 成功尝试记录：`outputs/results/candidate_retry/candidate_attempts.json`
- 单候选配置和失败记录：`outputs/results/candidate_baseline/`
- 对照摘要：`outputs/results/candidate_comparison.json`

输出路径沿用仓库规则，不纳入 Git；报告和代码改动保留在工作区，未提交或推送。

## 边界及后续方向

候选目前手工给定，采用第一个完整可行解，不比较所有候选代价。仍是固定任务序列下的候选站位重试，不是联合优化所有语义分支；抓取面变化、自动候选生成和命令事件日志尚未加入。

候选耗尽仅意味着当前列表与求解设置未找到通过验证的解。非凸求解器、有限凸覆盖和离散采样都限制了可作出的结论。下一步可增加独立场景变体及候选生成策略，比较固定站位与重试的成功率；区域可靠性问题应独立继续处理。
