# Sparse semantic plan refinement: doorway reach diagnostic

Date: 2026-10-10

## Question

Can the doorway task's sparse semantic milestones be refined into a continuous
collision-free C-space path, and what happens when the convex cover receives
too little intermediate structure?

This is a diagnostic of the existing Drake algorithms, not a new planner. The
manipulation runs use Drake IRIS-NP2 to grow collision-free convex sets and
Drake `GcsTrajectoryOptimization` with Clarabel to select a continuous path.
The core planning implementation was not changed for this experiment.

## Semantic task and encoding

The tested reach phase fixes the mobile base at `(-0.45, 0.50, 0)` in Room 1;
the free cup stays on the Room 2 table. Its arm endpoints are the tucked
navigation posture and the IK grasp posture. The full doorway task has these
milestones:

1. `base_in_room1`: the base collision geometry remains on the Room 1 side of
   the doorway plane, with the configured clearance. The base-navigation
   corridor and fixed staging pose encode this requirement in the GCS regions.
2. `end_effector_in_room2`: the gripper reaches the cup grasp pose on the Room 2
   side. In this implementation that is imposed by the IK goal and reach-phase
   endpoint; the trace also records the end-effector signed position relative
   to the doorway plane.
3. `arm_crosses_door`: at least one arm link passes through the doorway
   aperture. Collision checking against the wall and frame protects the path;
   the link-crossing predicate is independently audited from the executed
   geometry. It is not yet a separate semantic vertex in the GCS graph.
4. `cup_grasped`: a bilateral-grasp action vertex joins the free-cup reach
   layer to the attached-cup layer. SAP execution validates bilateral finger
   contact; it does not teleport or attach the cup.
5. `cup_retracted_to_room1`: the attached-cup GCS layer ends with the cup fully
   across the doorway plane, and the physical trace checks that it remains in
   contact while crossing.

The one-shot full-task graph already contains base navigation, free-cup reach,
the grasp action, and attached-cup retraction. This diagnostic isolates the
free-cup reach layer to study how sparse C-space guide samples affect region
connectivity.

## Experiment

For the same fixed scene, start posture, grasp pose, collision model, IRIS-NP2
settings, and GCS solver, we supplied 2, 5, 9, or 17 equally spaced samples on
the straight joint-space interpolation between the two reach endpoints. These
samples were only seeds for IRIS-NP2 region generation; the final path was
chosen by GCS. The existing bounded bridge mechanism was left enabled (up to
8 additional regions), and successful cases used the official convex
relaxation and rounding path solve.

| Input C-space guide samples | IRIS seeds selected | Result | Regions | Added bridges | GCS path collision audit |
|---:|---:|---|---:|---:|---|
| 2 | 2 | Disconnected; no unprovided reference state remained to insert | — | — | No path |
| 5 | 3 | Connected and GCS solved | 5 | 2 | 0 / 28 |
| 9 | 4 | Disconnected; the bounded routine exhausted usable reference candidates | — | — | No path |
| 17 | 5 | Connected and GCS solved | 5 | 0 | 0 / 28 |

The successful 5-sample case took 0.013 s for GCS; its full IRIS + GCS
diagnostic took 3.87 s. The 17-sample case took 0.018 s for GCS and 2.01 s
overall. Both were continuous 10-D phase trajectories in the planner's
`(x, y, yaw, q_arm)` state, with the base fixed by endpoint/reference
constraints. Their 1,000-sample per-region holdout audits had 0 collisions in
the 5-sample case and 1 in the 17-sample case; these are probabilistic region
audits, not formal certificates. The successful trajectory samples themselves
had zero collisions in both cases.

As an independent reachability diagnostic, direct joint-space interpolation
from navigation posture to grasp posture was checked at 257 configurations in
the free-cup collision model: 0 collisions. Direct interpolation from the
grasp state to the already-used Room 1 retraction endpoint was also checked at
257 configurations in the attached-cup collision model: 0 collisions. This
shows that the particular scene admits a simple continuous refinement; it
does not establish that arbitrary sparse semantic plans are refinable.

## Interpretation

This result illustrates the professor's point in a useful, bounded way:

- The semantic milestones are valid, but the two endpoint-centered local
  convex regions are disconnected in the generated cover. The current
  manipulation region domain uses a per-joint radius of 0.30 rad; the endpoint
  joint vectors differ by 1.610 rad in one coordinate, so their domain boxes
  cannot overlap (the difference exceeds the combined 0.60 rad radius).
- The collision-free C-space itself is not disconnected in this example: a
  direct interpolation passes the sampled collision audit. Thus the observed
  failure is a disconnected *convex cover/graph*, not evidence of a true
  topological disconnection of free configuration space.
- With 5 guide samples, the existing procedure can use two remaining
  collision-free references to grow bridge regions, after which the full GCS
  phase solve succeeds. With only the two semantic endpoints it has no
  candidate bridge state, so it fails. The 9-sample failure also shows the
  present greedy bridge heuristic is not monotonic in input density.
- The current method does not infer the missing motion structure from semantic
  predicates alone. It requires C-space reference samples; automatic
  discovery of the right intermediate state is still open.

Therefore, for this simplest doorway geometry we found continuous trajectories
and did not need post-processing after a connected cover was available. The
exploration does expose a concrete limitation: sparse endpoints alone do not
give the current cover builder enough information to repair the graph. The
The relevant follow-on is to compare a published GCS-connectivity repair
method rather than inventing an ad hoc midpoint rule. We ran the first
paper-based DirectLine + binary-inflation reproduction successfully; see
[report 59](59_gcs_bridging_paper_reproduction.md).

## Research connection

A directly relevant paper appeared on arXiv on 2026-08-23:
[GCS-Bridging: Restoring Connectivity of Disconnected Convex Sets for
Graph-of-Convex-Sets Motion Planning](https://arxiv.org/abs/2608.22326). It
formalizes the same graph-level failure: start and goal convex regions lie in
different connected components of the initial GCS map. Its method ranks
cross-component region pairs, tries a collision-checked DirectLine and/or
local RRT-Connect bridge between witness points, then inflates regions along
the found path using binary midpoint ordering until region-overlap connectivity
is restored. The authors report 99.8% reconnection for the RRT-C variant and
100% for their hybrid in their tested scenarios; those are paper results, not
results from our simulator. They also state that the method is budgeted and
not complete, and that IRIS-NP/NP2 regions are not rigorously collision-free
certificates.

This is more directly aligned with the professor's “insert missing structure”
question than our current bridge heuristic. Our current `bridged` mechanism
can only inflate additional configurations already present in the reference
sequence. It does not rank disconnected region pairs, search between their
witness points with LocalRRTC, or use binary inflation to add only the shortest
overlap chain. We should reproduce that published repair strategy as the next
method, preserve the semantic modes as hard boundaries, and compare:

1. sparse semantic endpoints with no repair (expected disconnected-cover
   failure);
2. current reference-sample bridge procedure;
3. GCS-Bridging with DirectLine, LocalRRTC, and their hybrid option.

For each case, record initial/final connected components, successful repair
rate over fixed seeds, added regions, repair and GCS solve time, independent
trajectory collision audit, semantic trace, and (for the final scenario) SAP
execution. This gives a fair test of whether the added method repairs the
semantic-to-C-space connectivity gap. It does not by itself solve the broader
problem of inferring task semantics from language.

## Limitations

- The direct interpolation checks are finite samples, not a continuous-time
  proof.
- This is one fixed base posture, one cup placement, and one grasp solution.
- The sparse-density sweep is diagnostic, not a statistical success-rate
  comparison.
- `arm_crosses_door` is audited from geometry but has not yet been made its own
  explicit GCS semantic layer.
- The experiment covers the reach phase; the already-generated full-task
  one-shot GCS trajectory and SAP execution are separate evidence.
