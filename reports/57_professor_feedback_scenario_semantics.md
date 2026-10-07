# 教授反馈整理：门口取物场景与 Semantic 设计

> 整理日期：2026-10-08
> 本文只整理教授的建议、问题含义与后续调研方向，不代表相关代码或实验已经修改。

## 1. 教授的原始建议

原始记录：

> 桌子在很近，手臂伸进去，base 在外面。一个 solution 中，Drake 将 base 放在门外，但是手臂还是可以伸进去。手臂从没进去。Semantic 很重要，这一步要好好设计，可以调研。

原始表述中“手臂伸进去”和“手臂从没进去”存在语义上的歧义，汇报或实施前需要再次向教授确认。本文暂不擅自把它们合并成一个确定要求。

## 2. 对建议的初步整理

教授关注的可能不是普通的“底盘从一个房间移动到另一个房间”，而是一个需要整机语义判断的门口取物场景：

1. 桌子或目标物体位于门内，并且距离门较近。
2. 移动底盘可以停在门外，不一定要穿过门。
3. 机械臂可能从门外伸过门口，进入另一个空间完成接近或抓取。
4. 因此，仅根据 base 的位置判断机器人“是否进入房间”是不够的。
5. GCS/Drake 即使找到了一条无碰撞解，这条解也不一定满足任务真正想表达的语义。
6. 需要认真定义 base、手臂、末端执行器和物体相对于门/房间的语义状态及允许的动作顺序。

核心提醒可以概括为：

> **几何可达和无碰撞不等于语义正确。场景必须迫使规划器处理“base 在外、arm 进入”的整机交互，而不能只展示一个底盘路径。**

## 3. 教授可能指出的当前场景问题

### 3.1 任务可能过于容易

如果杯子、桌子和门的位置使机器人在门外就能直接完成抓取，那么规划器不需要真正处理以下问题：

- 手臂什么时候可以穿过门口；
- 哪些 link 已经进入房间；
- base 是否允许留在门外；
- 抓取前后整机占用空间如何变化；
- 携物后是否还能安全撤出或通过门口。

这会让 semantic 层看起来存在，但实际上没有对解产生关键影响。

### 3.2 只看 base 会产生错误语义

假设用门所在平面将空间划分为 inside 和 outside。若只检查 base 的位置，可能得到：

\[
\operatorname{inside}(q_{base}) = \mathrm{false},
\]

但机械臂的某些 link 或末端执行器已经进入门内：

\[
\operatorname{inside}\bigl(p_{ee}(q)\bigr) = \mathrm{true}.
\]

因此下面两个状态不能被视为相同：

- base 在门外，机械臂也完全在门外；
- base 在门外，但机械臂或末端执行器已经伸入门内。

当前 semantic 状态若主要由 base 所在区域决定，就可能无法区分这两种情况。

### 3.3 “Drake 有一个 solution”不足以证明场景设计正确

Drake 返回可行解，只能说明在当前模型和约束下找到了满足条件的轨迹。它不能自动说明：

- 任务语义符合教授想要的行为；
- 场景确实需要 full-arm semantic；
- 解没有利用场景布置中的捷径；
- 动作顺序是正确的；
- 规划器真正区分了 base motion 和 arm-reaching motion。

所以需要检查“为什么这个解可行”，而不能只汇报“Drake 找到了一条解”。

## 4. 建议明确的 Semantic 谓词

后续可以围绕以下谓词调研和设计。这里是问题定义草案，不代表已经实现。

### 4.1 Base 谓词

\[
P_{base\_outside}(q) := p_{base}(q) \in \mathcal R_{outside}
\]

表示底盘位于门外允许区域。

### 4.2 末端执行器进入谓词

\[
P_{ee\_inside}(q) := p_{ee}(q) \in \mathcal R_{inside}
\]

表示末端执行器已经通过门口并进入门内工作空间。

### 4.3 全臂穿门安全谓词

\[
P_{arm\_door\_safe}(q)
:=
\bigwedge_{\ell\in\mathcal L_{arm}}
d\!\left(\mathcal G_\ell(q),\mathcal G_{door}\right) \ge \delta,
\]

其中 \(\mathcal G_\ell(q)\) 是第 \(\ell\) 个机械臂 link 的几何体，\(\mathcal G_{door}\) 是门框/墙体碰撞几何，\(\delta\) 是安全间隙。

### 4.4 可抓取谓词

\[
P_{graspable}(q)
:=
P_{base\_outside}(q)
\land P_{ee\_inside}(q)
\land P_{arm\_door\_safe}(q)
\land P_{grasp\_pose}(q).
\]

这个定义表达了一个可能的目标语义：base 保持在门外，但手臂安全伸入门内并到达合法抓取位姿。

### 4.5 动作顺序

一个可供讨论的动作序列是：

```text
approach_door
→ stop_base_outside
→ extend_arm_through_door
→ reach_pregrasp
→ grasp
→ retract_arm_with_object
→ resume_base_motion / place_object
```

其中 `stop_base_outside → extend_arm_through_door` 是教授反馈最可能希望重点展示的语义切换。

## 5. 场景应如何重新审视

在改代码前，应先画出并核对以下几何关系：

1. 门平面和 inside/outside 的明确定义；
2. 桌子及杯子相对门的距离；
3. base 停在门外时机械臂的最大可达范围；
4. 手臂伸入时各 link 与门框、墙和桌面的间隙；
5. 抓住杯子后，手臂能否带着杯子安全撤回；
6. 是否存在不触发预期 semantic 的更短捷径；
7. 当前场景到底要求 base 穿门、禁止 base 穿门，还是允许规划器在两种策略中选择。

一个有说服力的场景应同时具备：

- **必要性**：不考虑 full-arm semantic 时会产生错误或任务不完整的解；
- **可行性**：加入正确 semantic 后仍存在至少一条可执行解；
- **可辨识性**：从动画和数据中能清楚看出 base 在外、arm/EE 在内；
- **对照性**：能够比较 base-only 与 semantic-aware 两种规划结果。

## 6. 建议的对照实验

### A. Base-only 对照

- 只使用 base 的房间/门口区域谓词；
- 记录规划器返回的 base 和 arm 轨迹；
- 检查是否出现语义上不符合预期、但几何上无碰撞的解。

### B. Full-arm semantic

- 加入 `base_outside`、`ee_inside`、`arm_door_safe` 和 `graspable`；
- 使用同一场景、同一起点和目标；
- 检查语义约束是否真正改变了解或排除了错误解。

### C. 场景位置消融

至少比较三种预先设定的位置：

1. 桌子离门太近：可能不需要手臂真正进入；
2. 合理距离：base 在外、手臂必须伸入；
3. 桌子离门太远：base 在外时不可达，应被正确判为不可行或要求 base 进入。

需要在实验前固定位置和判定标准，避免根据结果反复调整场景。

## 7. 需要向教授确认的问题

1. “手臂从没进去”具体是描述当前系统的问题，还是要求手臂不允许进入？
2. 教授希望 base 必须留在门外，还是允许 GCS 在“base 穿门”和“arm 伸入”之间选择？
3. “一个 solution”指单条 Drake 可行轨迹，还是希望整个 long-horizon task 一次联合求解？
4. 抓取目标是在门内、桌面上，还是门口边缘？
5. Semantic 应约束末端执行器、任一 arm link，还是整个机械臂包络？
6. 教授希望的关键对照是 base-only vs full-arm semantic，还是普通 sequential pipeline vs one-shot GCS？

## 8. 当前结论

教授的反馈说明，现有演示场景可能没有充分暴露 full-arm semantic 的必要性。下一步不应先追求更漂亮的动画，而应先确定一个没有捷径的门口取物场景，并给出可验证的谓词、动作顺序和 base-only 对照。

在上述歧义得到确认前，不应声称当前场景已经证明：

- semantic 已经统一进入 long-horizon GCS；
- base 和 arm 的动作顺序已经被一次联合优化；
- 门口取物场景已经正确体现教授提出的问题。
