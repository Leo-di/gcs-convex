# Semantic constraints at the IRIS interface

For the new typed `Inside` DSL and **both conservative-domain and direct-nonlinear interfaces**, start with [Inside: one predicate, two IRIS interfaces](inside_compiler.md). The description below covers the earlier cafe and room-domain implementation.

In that implementation, the semantic check runs before IRIS. It produces a bounded configuration-space domain. Drake's IRIS-NP2 receives that domain and a geometric collision checker, then returns a polyhedron inside the domain.

## Reading order

| Function | File | What to inspect |
|---|---|---|
| `operation_oracle` | [`green_room_gcs.py`](../src/green_room_gcs.py) | Selects room containment alone or the combined spatial evaluator. |
| `OperationSpatialOracle.query` | [`robot_keep_clear.py`](../src/robot_keep_clear.py) | Combines room, reserved-space, and enabled human-access residuals. |
| `semantic_box_bound` | [`green_room_gcs.py`](../src/green_room_gcs.py) | Converts center residuals into a lower bound over a seven-joint box. |
| `make_operation_cover` | [`green_room_gcs.py`](../src/green_room_gcs.py) | Reduces padding and subdivides guide intervals until the bound passes. |
| `grow_cover` | [`green_room_iris.py`](../src/green_room_iris.py) | Supplies the domain, seed, collision checker, and options to the IRIS wrapper. |
| `solve_capped_iris` | [`iris_conditioning.py`](../src/iris_conditioning.py) | Calls official `IrisNp2`; applies a similarity-coordinate retry for a specific near-support numerical failure. |
| `build_audited_cover` | [`operation_refinement.py`](../src/operation_refinement.py) | Checks regions and refines rejected domains. |
| `solve_operation` | [`green_room_gcs.py`](../src/green_room_gcs.py) | Builds the region graph and calls seven-joint GCS. |

## The fixed-base calculation

For arm coordinates \(a\in\mathbb R^7\), let the local box be \(B=[l,u]\), its center \(c=(l+u)/2\), and half widths \(h=(u-l)/2\). The base and fingers are fixed within this query.

The evaluator returns separate residuals for the base and for the moving arm/gripper geometry. Positive values mean that the modeled requirement is satisfied. With reach bound \(R\), the implemented box bound is

\[
\underline\rho(B)=\min\left(\rho_{base}(c),\rho_{arm}(c)-R\sum_{j=1}^{7}h_j\right).
\]

`arm_point_reach_bound` includes frame translations and collision-shape extents. When human-access checking is enabled, it also accounts for the body AABBs used by that evaluator and the payload reach where applicable. The base term remains unchanged because the base is fixed.

`make_operation_cover` clips each guide interval to joint limits, allocates padding from the remaining arm-motion allowance, and checks this bound. If it is below `1e-6`, it bisects the guide interval and retries. The recursion is bounded; exhaustion raises an error for this local cover.

## What IRIS receives

`grow_cover` calls `solve_capped_iris`, whose unconditioned call is equivalent to:

```python
region = IrisNp2(checker, starting_ellipsoid, domain, options)
```

- `domain` is the `HPolyhedron` produced by the preceding spatial checks.
- `checker` evaluates robot/environment and self collision in Drake.
- `options.parameterization` maps the seven arm coordinates into the full plant positions, retaining the fixed base and fingers. It supports both doubles and AutoDiff.
- `starting_ellipsoid` supplies the local seed.
- `region.A()` and `region.b()` define the returned set, `A @ a <= b`.

For near-support numerical failures, the wrapper can transform the domain, seed, margin, and parameterization into similarity coordinates, retry official IRIS, and transform the result back. The existing implementation leaves Drake's C++ IRIS algorithm unchanged. Human access is evaluated by project code; it is not added as a natural-language input or a direct connectivity constraint inside IRIS.

For human access, [`human_passage.py`](../src/human_passage.py) uses vertical capsules, conservative body AABBs, a guarded grid, and four-neighbor routes. The domain bound preserves a route witness under bounded occupation changes. This model assumes static furniture and evaluates instantaneous access.

## Inspect one domain

From the repository root:

```bash
LCM_DEFAULT_URL=memq:// python scripts/inspect_semantic_iris.py \
  --run-iris --output outputs/semantic_iris_example
```

This uses the primitive doorway scene and Drake's bundled robot models. It constructs a small joint perturbation at a fixed base; it is a domain-inspection example, not a grasp or task-completion experiment.

The checked example is saved in [`examples/semantic_iris_inspection.json`](examples/semantic_iris_inspection.json). With a requested padding of `0.03 rad`, the semantic-domain constructor selected approximately `0.015123 rad` and reported a lower bound of `0.231428 m`. IRIS returned a seven-dimensional region with 14 faces. In this example, the input box was already small enough that no additional collision faces were needed.

The JSON includes the guide, input-domain bounds, residual terms, and output matrices. Omit `--run-iris` to inspect domain construction alone.

## Joint base–arm extension

[`whole_task_joint_gcs.py`](../src/whole_task_joint_gcs.py) uses ten coordinates: base x/y/yaw and seven arm joints. Its domain bound includes base translation and rotation. [`semantic_domain_expansion.py`](../src/semantic_domain_expansion.py) allocates padding using the remaining spatial slack.

The whole-task graph shares grasp/place configurations across motion modes. Exact FK and near-support constraints enter the nonlinear restriction problems. Near-support domains are recorded separately from IRIS regions.

The newer cold-search and base-clearance extensions remain experimental. Recent larger-domain runs failed nonlinear restriction, execution, or conservative-domain admission checks. A previous fixed-order whole-task run passed execution, but that result does not validate all subsequent extensions.

## Validation limits

IRIS-NP2 uses sampling and nonlinear counterexample search. The code also checks region samples, interpolated references, and measured execution. These checks have different scopes. Region membership alone does not establish collision safety, and successful sampled human-access checks do not model a pedestrian moving through the scene.
