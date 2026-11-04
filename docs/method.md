# Method

## 1. State, inputs, and output

| Symbol | Definition |
|---|---|
| \(q=(b,a)\) | Base pose \(b=(x,y,\psi)\) and arm coordinates \(a\in\mathbb R^7\). |
| \(X\) | Object poses and support relations. |
| \(m\) | Empty-hand, carrying, or contact mode. |
| \(S=(q,X,m,C)\) | Planning state; \(C\) records completed actions. |
| \(\mathcal A\) | Finite docking, grasp, transport-posture, and placement candidates. |
| \(\prec\) | Required action precedence. |
| \(\Phi\) | Structured spatial rules, including entity scope, activation, and margin. |
| \(D_i,P_i\) | Local input domain and returned IRIS polyhedron. |

**Input:** robot and scene models, initial state, object-placement goals, \(\mathcal A\), \(\prec\), \(\Phi\), and a planning budget.

**Output:** selected actions and a full robot trajectory, or a recorded planning failure. Execution produces a separate measured trace and acceptance result. Gripper opening/closing profiles are prescribed in the present implementation.

## 2. Task search and geometric guides

Search maintains the complete object state after each candidate placement. Eligible actions respect \(\prec\). A fixed order is supported for controlled comparisons; free-order search uses a distance heuristic to rank eligible actions. The heuristic does not establish order optimality.

IK and staged motion planning generate guide configurations for a complete action candidate. The joint optimizer searches regions around these guides. In the coupled implementation, rejection of the complete joint trajectory returns to task search.

## 3. Semantic input domains

For a guide interval, construct \(D=[l,u]\), with center \(c\) and half widths \(h\). The spatial evaluator provides center residuals and, for human access, a feasible route witness. A positive residual denotes remaining clearance under the rule's geometric model.

For fixed-base arm planning, the implemented sufficient domain condition is

\[
\underline\rho(D)=\min\{\rho_{base}(c),\rho_{arm}(c)-R\|h\|_1\}\geq\eta.
\]

For joint base–arm planning, the displacement bound includes translation and rotation:

\[
\Delta(D)=\sqrt{h_x^2+h_y^2}+R\sum_{j=3}^{10}h_j.
\]

The spatial rule is evaluated before IRIS. Padding is reduced or the guide interval is bisected when its conservative domain bound fails. The witness, geometric approximation, fixed world state, and motion bound determine the scope of this condition.

```text
BuildRegions(S, guide, mode):
    for each guide interval (qa, qb):
        D <- joint-limit-clipped local domain around (qa, qb)
        evaluate center residuals and access witness
        reduce padding or subdivide until the domain bound passes
        P <- IrisNp2(collision_checker(S, mode), seed, D, options)
        check region samples; refine rejected domains within budget
        retain P, its domain, and check results
    return retained regions
```

For the fixed-base implementation, the exact interface is traced in [semantic_iris.md](semantic_iris.md). The staged transport baseline uses separate two-dimensional base regions. The joint extension uses ten-dimensional IRIS regions away from support contact.

## 4. Task-wide region graph

Each motion vertex contains a convex region and trajectory control points. Edges require region overlap and compatible manipulation modes. For the current first-order representation,

\[
q_i(s)=(1-s)c_{i,0}+s c_{i,1},\quad c_{i,0},c_{i,1}\in P_i.
\]

Adjacent modes share their boundary configuration. A contact vertex constrains the gripper target pose through FK, leaving the robot configuration variable. The local contact approximation is used in the relaxation; exact FK and explicit near-support collision constraints are enforced in the restriction problems. Support-contact domains are recorded separately from IRIS output.

The objective combines weighted path energy, region-duration cost, and a small guide regularizer. Region durations are currently fixed individually; execution is retimed afterward.

## 5. Solve and admit

```text
SearchTask(S0, goals):
    select an eligible action and grounding candidates
    generate guides and propagate object/support states
    once a complete action candidate is available:
        construct regions and the shared whole-task graph
        solve the convex relaxation with Clarabel
        sample candidate graph paths from the relaxed flows
        solve each full-task nonlinear restriction with Ipopt
        check the selected reference independently
        accept, or return to action/candidate search
```

The current extension considers up to five rounded paths. A successful restriction is a feasible local solution for its selected path, not a global-optimality result.

## 6. Execution and comparison

Execute the accepted reference in one continuous free-body contact simulation. Check grasp contact, unsupported transport, settled placement, unexpected contact, and spatial rules. Recompute human-access connectivity independently from measured states.

The primary comparison is early semantic-domain construction versus post-hoc checking with repair/retry. Both receive the same rules, candidates, geometry, clearance margins, motion backend, and total budget. First fix the action order; evaluate order search separately. Include guide generation, failed candidates, region checks, and all restriction solves in timing.

**Implementation status.** The fixed-base semantic-domain/IRIS interface is implemented and has a small runnable inspection example. A fixed-order whole-task experiment passed contact execution. Cold task search, larger domains, and the conservative base-clearance extension are under development; their full matched evaluation is incomplete. Drake's IRIS and GCS C++ cores are unchanged.

**For discussion:** Should the main contribution be semantic-domain construction, contact-transition admission, or a change within IRIS? Which spatial-rule model and conservatism are acceptable for the first study?
