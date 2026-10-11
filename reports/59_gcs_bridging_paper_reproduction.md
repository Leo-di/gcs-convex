# Paper-based GCS connectivity repair: doorway arm reach

Date: 2026-10-10

## Result

We reproduced the DirectLine case of the published GCS-Bridging procedure on
the sparse doorway reach task. Two endpoint IRIS-NP2 regions began as two
separate graph components. The closest-point DirectLine passed Drake's edge
collision check. Binary path inflation added three intermediate IRIS-NP2
regions; the overlap graph then became connected. The shortest connected chain
retained two of those three bridge regions, and Drake GCS solved a continuous
10-D trajectory through that chain.

The GCS trajectory passed 200 consecutive Drake edge-collision checks with no
collision. The base stayed in the Room 1 staging set, its computed minimum
footprint clearance to the doorway plane was 0.150 m, and the end effector
entered Room 2 to a maximum depth of 0.461 m. Each of the five generated
IRIS-NP2 regions had 200 independent holdout samples with zero observed
collisions. These are finite-sample audits, not a continuous-time collision
certificate. This is a reach-phase result; this new path was not run through
SAP, and LocalRRTC was not invoked because DirectLine succeeded.

## Method

The experiment follows the relevant stages in [GCS-Bridging (arXiv:2608.22326)](https://arxiv.org/abs/2608.22326):

1. Grow an IRIS-NP2 region around each semantic reach endpoint and compute the
   initial overlap graph.
2. For the disconnected start/goal components, find the closest pair of
   configuration-space witness points with a convex quadratic program.
3. Validate the witnesses and check the line segment with
   `SceneGraphCollisionChecker.CheckEdgeCollisionFree`.
4. Since the direct line is valid, use it as the bridge path. Inflate an
   IRIS-NP2 region at the path midpoint; if connectivity is still absent,
   subdivide the longest remaining path interval and inflate its midpoint.
5. Once a connected chain exists, retain its shortest overlap path and call
   Drake `GcsTrajectoryOptimization.SolvePath` once with Clarabel convex
   relaxation and rounding.
6. Independently check the final GCS trajectory edge by edge against the same
   mode-specific collision checker.

The state is `(x, y, yaw, q_arm[0:7])`. The base staging set is restricted to
`x <= -0.45 m`; this preserves Room 1 membership and the minimum footprint
clearance for the selected base geometry. The free cup remains a collision
obstacle throughout the arm-reach mode. The grasp transition and attached-cup
retraction are outside this local connectivity experiment.

## Measurements

| Measurement | Result |
|---|---:|
| Initial IRIS regions / connected components | 2 / 2 |
| Closest cross-component witness gap | 1.550 (weighted C-space units) |
| DirectLine collision check | Passed |
| LocalRRTC | Not invoked |
| Binary inflation attempts | 3 |
| Final graph | 5 regions, 1 connected component |
| Shortest overlap chain used by GCS | 4 regions (2 original + 2 bridge regions) |
| Initial IRIS time | 7.91 s |
| Total time before GCS | 9.05 s |
| GCS solve time | 0.013 s |
| GCS trajectory edge audit | 0 / 200 collisions |
| IRIS holdout audit | 0 / 1,000 collisions |
| Base minimum doorway-plane footprint clearance | 0.150 m |
| Maximum end-effector depth into Room 2 | 0.461 m |

The exact machine-readable record is
[`result.json`](../outputs/results/sparse_gcs_bridging_v1/result.json), and
the experiment implementation is
[`experiment_gcs_bridging.py`](../src/experiment_gcs_bridging.py). Reproduce
with:

```bash
LCM_DEFAULT_URL=memq:// .venv/bin/python -m src.experiment_gcs_bridging
```

## What this says about the professor's question

This is a concrete example of a sparse task-level plan whose endpoint regions
do not initially form a connected GCS map. A collision-free path exists, and
the missing intermediate motion structure can be inserted as collision-free
convex regions. In this simple scene, DirectLine was enough; a harder geometry
would need the LocalRRTC branch to find a nonstraight bridge before inflating
regions along it.

The experiment is not yet a complete solution to semantic-plan refinement.
The repair occurs within one fixed semantic mode, and the mode sequence
(`base in Room 1 -> arm reach -> grasp -> retract`) is supplied. We have not
shown automatic discovery of omitted semantic modes, nor have we compared
DirectLine, LocalRRTC, and hybrid repair over randomized doorway scenes. Those
are the next proper experiments. The previous sparse-density diagnostics are
in [report 58](58_sparse_semantic_refinement_connectivity.md).

## Current evidence limits

- `CheckEdgeCollisionFree` uses the configured collision-checking edge
  resolution; the 200-segment trajectory audit is dense numerical evidence,
  not a formal continuous-time proof.
- IRIS-NP2 uses probabilistic region generation; zero collisions in 200 holdout
  samples per region does not establish that the entire polytope is collision
  free.
- This run verifies kinematic reach-phase path feasibility, not SAP execution
  or bilateral physical grasp. The earlier full-task physical run remains
  separate evidence.
- The paper is an arXiv preprint. Its reported aggregate success rates are
  literature results, not results from this experiment.
