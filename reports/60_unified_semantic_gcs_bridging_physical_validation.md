# Unified semantic GCS with sparse-reach connectivity repair

Date: 2026-10-10

## Result

The sparse-reach connectivity repair is now integrated into the full doorway
task graph. One Drake `GcsTrajectoryOptimization.SolvePath` call with Clarabel
selected a continuous trajectory over the complete sequence

`Room 1 base navigation -> free-cup arm reach -> bilateral grasp -> attached-cup lift and retraction to Room 1`.

The SAP execution passed the task audit. The base stayed in Room 1 with at
least 0.146 m geometric clearance to the doorway plane; the end effector
entered Room 2 by up to 0.499 m; an arm link crossed the doorway; the cup was
grasped with bilateral finger contact and carried back into Room 1. There were
zero unexpected contacts. The free-cup trajectory and the attached-cup
trajectory each passed their independent Drake collision checks at all 142
stored configurations.

## What was added

The free-cup reach layer begins with two sparse endpoint IRIS-NP2 regions and
two disconnected graph components. The closest cross-component witnesses had
a weighted configuration-space gap of 1.550. Drake's collision checker
accepted their direct segment. Three intermediate IRIS-NP2 inflations were
then attempted; the third connected the cover. The shortest overlap chain used
four regions (both endpoint regions and two bridge regions).

These connected reach regions were added to the same global GCS graph as the
Room 1 base-navigation region, the bilateral-grasp action vertex, and the
attached-cup retraction cover. The planner's selected path crossed every
semantic layer. No separate reach trajectory was stitched into the result.
The state was `(x, y, q_arm[0:7])`; base yaw was fixed at zero and removed
exactly. The continuous trajectory was selected by one rounded convex GCS
solve. Region construction took 17.31 s in total; the global GCS solve took
0.079 s.

The full SAP simulation used a free cup and actual bilateral finger contacts;
the cup was not attached or teleported. It recorded a maximum finger contact
force of 43.1 N, a minimum bilateral force of 18.3 N while carrying, and a
maximum relative cup-to-gripper translation change of 0.80 mm. The cup stayed
on the pickup table during the closing action. At the start of the first
attached-mode segment, table support persisted for about 50 ms while the cup
lifted. That existing segment is now labeled `lift_cup` in the execution
schedule so the contact audit recognizes the expected support-to-lift
transition; no waypoint or GCS constraint was added.

## Measurements

| Measurement | Result |
|---|---:|
| Global semantic GCS solves | 1 |
| Solver / optimization state | Clarabel / 9-D `(x,y,q_arm)` |
| Region counts: navigation / free-cup reach / grasp / attached-cup | 1 / 4 / 1 / 8 |
| Sparse reach graph before repair | 2 regions, 2 components |
| Bridge method and added regions | DirectLine + binary IRIS-NP2, 3 added |
| Final sparse-reach overlap graph | 5 regions, 1 component |
| Global GCS solve time | 0.079 s |
| Region-generation time | 17.31 s |
| Stored trajectory collision audit | 0 / 142 collisions |
| Minimum base-to-doorway footprint clearance | 0.146 m |
| Maximum end-effector depth in Room 2 | 0.499 m |
| Bilateral grasp / cup retracted to Room 1 | Pass / pass |
| SAP unexpected contacts | 0 |
| SAP max relative translation during carry | 0.80 mm |

## Limits

The attached-cup IRIS-NP2 cover had 3 colliding configurations in 1,600
independent holdout samples. The selected final trajectory itself passed its
142-configuration collision audit and its SAP execution, but this finite audit
shows the cover still needs tightening before treating every point in each
polytope as certified. The collision audits are discrete numerical checks,
not a formal continuous-time safety proof. The planar base remains an ideal
fully actuated model rather than wheel dynamics.

This experiment demonstrates that sparse semantic endpoint regions can be
disconnected in C-space, and that adding collision-checked intermediate
convex regions can make a complete semantic GCS trajectory executable in this
scene. It does not yet automatically infer omitted semantic predicates or
semantic modes. DirectLine succeeded here, so LocalRRTC was not exercised.

## Reproduction and artifacts

Run from the project root:

```bash
LCM_DEFAULT_URL=memq:// .venv/bin/python -m src.animate_unified_doorway_gcs \
  --reach-method gcs_bridging \
  --output outputs/results/doorway_arm_reach_semantic_v13/gcs_bridged
```

The main artifacts are [`result.json`](../outputs/results/doorway_arm_reach_semantic_v13/gcs_bridged/result.json),
[`semantic_trace.json`](../outputs/results/doorway_arm_reach_semantic_v13/gcs_bridged/semantic_trace.json),
the [3D Meshcat recording](../outputs/results/doorway_arm_reach_semantic_v13/gcs_bridged/scene.html),
and the [review page](../outputs/results/doorway_arm_reach_semantic_v13/review.html).
The implementation is [`animate_unified_doorway_gcs.py`](../src/animate_unified_doorway_gcs.py)
and the region-bridge routine is [`experiment_gcs_bridging.py`](../src/experiment_gcs_bridging.py).

The connectivity-repair stages follow the method in the
[GCS-Bridging preprint](https://arxiv.org/abs/2608.22326). This run's numbers
are our reproduction results, not the paper's aggregate results.
