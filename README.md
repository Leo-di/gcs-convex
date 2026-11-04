# Semantic mobile manipulation with Drake

Research code for constructing spatially constrained IRIS regions and planning mobile manipulation with GCS. Rules are structured inputs; the current implementation uses Drake's IRIS-NP2 and GCS APIs without changes to their C++ cores.

## Start here

See [Inside in the complete cafe task](docs/inside_cafe.md) for the ten-coordinate base/arm compiler, three-cup task graph, both IRIS interfaces, contact constraints, measured execution checks, and a portable common guide.

0. [Inside DSL → compiler → IRIS → GCS](docs/inside_compiler.md): runnable seven-joint comparison of conservative input domains and direct nonlinear semantic constraints, with measured results and actual region matrices.
1. [Semantic constraints at the IRIS interface](docs/semantic_iris.md): the calculation, source-code reading order, and a runnable example.
2. [Example input domains and IRIS matrices](docs/examples/semantic_iris_inspection.json).

The Inside compiler now exposes two interfaces:

```text
Inside(robot, work_area) -> FK inequalities g(q) <= 0
    A: conservative whole-box bound -> restricted IRIS domain
    B: MathematicalProgram(g) -> IRIS additional nonlinear constraints
    -> IRIS polyhedra -> intersection graph -> GCS -> independent checks
```

The earlier cafe/whole-task implementation uses conservative spatial domains:

```text
structured spatial rule + scene + guide configurations
    -> conservative local configuration-space domain
    -> Drake IrisNp2(collision_checker, ellipsoid, domain, options)
    -> convex regions Aq <= b
    -> GCS trajectory planning
    -> reference checks and contact execution
```

## Installation

The review environment uses Python 3.13 and Drake 1.57.0. Run commands from the repository root.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Inspect the semantic-to-IRIS call

For the complete new Inside comparison:

```bash
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/run_inside_iris_comparison.py --output outputs/inside_run
```

This is fixed-base arm reference planning. See the [compiler guide](docs/inside_compiler.md) for report/replay generation, reproduction settings, and limitations.

This small example uses primitive scene geometry and Drake's bundled robot models. It needs no ReplicaCAD download.

```bash
LCM_DEFAULT_URL=memq:// python scripts/inspect_semantic_iris.py \
  --run-iris --output outputs/semantic_iris_example
```

The output records the guide, requested and accepted padding, center residuals, conservative box bound, and IRIS matrices. The example inspects one local joint perturbation at a fixed base; it does not execute a grasp.

Core regression checks:

```bash
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q \
  tests/test_robot_keep_clear.py tests/test_human_passage.py \
  tests/test_operation_refinement.py tests/test_semantic_domain_expansion.py \
  tests/test_task_order_search.py tests/test_whole_task_joint_gcs.py
```

## Source map

| Code | Responsibility |
|---|---|
| [`green_room_gcs.py`](src/green_room_gcs.py) | Fixed-base semantic box bounds, domain subdivision, and seven-joint GCS. |
| [`green_room_iris.py`](src/green_room_iris.py) | Collision-checker parameterization and the official IRIS-NP2 call. |
| [`robot_keep_clear.py`](src/robot_keep_clear.py) | Combined room, reserved-space, and human-access evaluator. |
| [`human_passage.py`](src/human_passage.py) | Capsule/grid access witnesses and clearance residuals. |
| [`operation_refinement.py`](src/operation_refinement.py) | Region checks and bounded domain refinement. |
| [`whole_task_joint_gcs.py`](src/whole_task_joint_gcs.py) | Joint base/arm mode graph and variable contact configurations. |
| [`cold_joint_task_search.py`](src/cold_joint_task_search.py) | Task search with feedback from full-trajectory admission. |
| [`whole_task_execution.py`](src/whole_task_execution.py) | Reference reconstruction, independent checks, and execution. |

## Whole-task experiments

The complete Inside comparison has its own entry point:

```bash
python scripts/prepare_inside_cafe_guide.py --output outputs/inside_cafe_guide
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/run_inside_cafe.py --guide-run outputs/inside_cafe_guide \
  --output outputs/inside_cafe_run --execute
```

Both branches build fresh covers from the same staged guide. Each final trajectory is obtained from one full-task 10D graph with variable contact configurations. Task order and finger profiles remain supplied. Cafe assets are external; the guide bundle contains configuration and reference data only.

A fixed-order three-cup development run passed continuous contact execution and independent sampled access checks. The subsequent cold-search, larger-domain, and base-clearance extensions are experimental. Recent failures include nonlinear restriction failure, execution contact, and conservative-domain rejection; a successful earlier run does not validate these extensions.

The cafe configuration uses external ReplicaCAD-derived assets. Asset preparation is provided in [`prepare_replicacad.py`](scripts/prepare_replicacad.py); meshes, recordings, and generated results are excluded from Git. The complete-task entry point, once the referenced assets are prepared, is:

```bash
LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/run_coupled_cafe.py \
  --task configs/semantic_service_cafe.json \
  --output outputs/cafe_run --method early --order-mode search --execute
```

Local IRIS covers, finite candidate search, sampled checks, and an ideal planar-base simulation define the current scope. Higher-order trajectory smoothing, continuous-time access guarantees, and hardware validation remain open.

Earlier experiment code and reports remain in the repository history and existing directories. The source guide above describes the review entry points and current limitations.
