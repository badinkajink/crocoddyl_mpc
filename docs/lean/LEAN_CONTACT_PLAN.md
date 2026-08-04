# Lean-Task Contact Selection — Multi-Session Working Brief

Created 2026-08-02 21:34 MDT. This file is the durable context for a multi-session
investigation. Sessions are spaced to fit usage-limit resets; each session appends
its results here.

## The idea (user's framing, 2026-08-02)

Allen produced an H12 lean where the robot braces against a table with the **elbow
link**. Observation: the three plausible bracing contacts map almost 1:1 onto
distinct actuators, so cost construction to elicit each form of contact should be
simple and separable:

| Contact site      | Driving joints                          |
|-------------------|-----------------------------------------|
| Elbow only        | shoulder roll + shoulder pitch          |
| Forearm           | + elbow pitch (from the elbow-braced pose) |
| Gripper "palm"    | + wrist yaw (gripper rotated 90° to brace on its side) |

Not proposing separate hand-written strategies — proposing that **contact selection
be optimized**, not scripted.

Hypothesized formulation (roughly an LP / cascaded optimization):

1. Target reach pose (XY reach, Z height) is known offline.
2. Back-compute where the lean must place the reaching arm / torso.
   Assumption: torso joint unused — only guaranteed if we lock it.
3. Analytically compute the loss of stability and the **restorative bracing wrench**
   required.
4. Required contacts are then governed by that wrench plus the reach kinematics:
   - near reach, moderate Z → small wrench → single **palm** brace (wrist yaw) suffices
   - near reach, very low Z → multiple contacts needed just to *lower* the reach
   - far reach → in principle **elbow** brace alone, but if the objective favors
     distributed load, triple contact (elbow + forearm + palm) should emerge naturally.
5. Deployment: solve offline (crocoddyl? or the cascaded scheme from the paper below),
   then continuously refine online with MPC.

Open question to answer first: is this actually feasible / well-posed, and where does
it break?

## Reference paper under review

`26-0767_01_MS_compressed.pdf` (repo root, currently on the `sampling` branch).
Simultaneous contact selection and planning via cascaded optimization. Even if the
exact method doesn't transfer, the grounding, math, and related work should. Our
problem is *easier* in the contact geometry (single flat table surface) and *harder*
in the plant (H12 humanoid, hard to control).

## Immediate deliverable the user asked for

> "can you create a chart showing joint torques at this sustained position?"

Joint-torque chart for the sustained braced lean pose from Allen's result.

## Branch situation

`git branch -a | grep -i icra` → nothing. Only remote configured is
`origin git@github.com:badinkajink/mujoco_mpc.git`. Allen's `icra2026` / `icra26`
branch is not fetchable yet — **need his remote/fork URL from the user** before
Session 1 can check it out. Current working branch is `sampling` with uncommitted
work (IDTO bug report, triple_pendulum_cartpole benchmark reports).

## Session plan

Ground rules the user set: be methodical; produce **plans and analysis, including
visual renders**, for everything tried; spread usage across sessions; write results
back into this file so the next session starts warm.

### Session 1 — ~00:35 MDT Aug 3 (first usage reset). Read & ground. No heavy compute.
- [ ] Get Allen's remote from the user; fetch and check out the `icra2026`/`icra26`
      branch (stash or leave `sampling` work alone — do not clobber it).
- [ ] Read the paper end to end. Extract: problem statement, decision variables,
      the cascade structure, contact-mode parameterization, what makes it tractable,
      what it assumes about the surface/contact set, and its stated failure modes.
- [ ] Map paper → our problem in a written table: what transfers, what doesn't,
      what our flat-table assumption buys us.
- [ ] Locate the lean task in the branch: task XML, cost terms, current contact
      handling, which joints are locked, whether the torso joint is locked.
- [ ] Produce the **joint-torque chart** for the sustained braced pose (the user's
      explicit ask). Use `planner-eyes` conventions: render the rollout, don't infer
      from the cost table.
- [ ] Append findings here. Stop.

### Session 2 — ~05:35 MDT Aug 3 (second reset). Formalize & prototype the math.
- [ ] Write the static-equilibrium / restorative-wrench derivation for the lean:
      given target reach (XY, Z) and support polygon, what wrench must the brace supply?
- [ ] Set it up as the smallest honest optimization (start LP/QP over contact wrenches
      at the three candidate sites with friction-cone constraints; contact *selection*
      via l1 / cardinality relaxation before reaching for anything fancier).
- [ ] Sanity-check numerically offline in Python against a handful of reach targets,
      and check the predicted regimes: near→palm, low-Z→multi-contact,
      far→elbow, load-distribution→triple contact.
- [ ] Visual output: reach-target map colored by predicted contact set, plus wrench
      magnitude. Append here.

### Session 3+ — Integrate.
- [ ] Turn the predicted contact set into MJPC cost terms / task variants and test
      whether the behavior actually emerges in sim.
- [ ] Renders + filmstrips per outcome, timestamped run dirs, provenance manifests,
      seed replicates, HTML report (per this user's standing experiment hygiene).
- [ ] Decide offline-solver question (crocoddyl vs. staying in MJPC) with evidence,
      not preference.

---

# SESSION 1 FINDINGS (2026-08-03)

## S1.1 Branch

`origin/icra2026` exists (229 commits ahead of `origin/main`); the Session-0 note
that it was missing was wrong. Checked out as a **git worktree** at
`/home/humanoid/Programs/mjpc_icra2026` so the `sampling` tree is untouched.

Build: configures and builds with one workaround —
`avoid.cc:702` trips `-Werror=unused-result` on an ignored `system()` return.
Configured with `-DCMAKE_CXX_FLAGS="-Wno-error=unused-result"` rather than editing
Allen's source. (Worth telling him: that file breaks a clean GCC build.)

## S1.2 What the lean task already is

Files: `mjpc/tasks/humanoid_bench/lean/{lean.h,lean.cc,Lean_H12*.xml,strategies/*.json}`.
`lean.cc` is 5.4k lines; 36 strategy slots; three model variants (base / Magpie
grippers / dexterous Hands).

**The critical finding for this idea:** contact selection is *already an explicit
data structure* — it is just hand-authored. Each phase in a strategy JSON carries

```json
"contacts": [ {"body1": 28, "body2": 0,
               "local_pos1": [0.17,0,0], "local_pos2": [0.8,-0.24,0.75]}, … ]   // up to 5
"brace_force_target": 70.0
```

i.e. **(robot body, point on that body) ↔ (world body, point on the table)** pairs
plus a scalar normal-force demand. In `h12_pipeline_forearm_brace`'s terminal phase
`forearm_brace_lean` the three active pairs are body **28** (hand), body **25**
(forearm/elbow), body **7** (foot). Those are consumed by the `Contact` residual
(dim 15 = 5 pairs × 3) at weight 1000, alongside `Brace Pos` (dim 3, w 80) and
`Brace Force` (dim 1, w 60).

So the optimizer you're describing has a **well-defined output slot already wired
into the cost**: it emits the `contacts` array + `brace_force_target`, and MJPC's
existing residual machinery executes it. That is a much shorter integration path
than I expected — no new cost term is strictly required for a first version.

Supporting machinery already present and reusable:
- **Phase ladder** with smoothstep ramps between phases (`kPhaseRampSeconds` 1.5 s,
  asymmetric ascent 3.0 s / descent 0.6 s) and per-phase weight blending
  (`ApplyRampedWeights`), so a *changed* contact set is introduced gradually rather
  than as a step — the paper's "avoid frequent contact switching" concern is
  already handled here by `contact_pair_is_new_[]` ramping each newly-appeared pair
  from 0 to full over 1.5 s.
- **Runtime target seam**: `Reach Active / Reach X/Y/Z` task parameters (indices 3–6)
  already accept a live (X,Y,Z) reach target over gRPC. **This is the natural input
  to the contact-selection solver** — reach target in, contact set out.
- `reach_hand=0` auto-picks the nearer arm; the bracing arm is defined as the other.

Design doc `docs/specs/2026-06-22-lean-pipeline-31-35-design.md` is the authority on
the intended pipeline and explicitly frames stages 31–35 as a hand-tuned ladder with
a generator script — exactly the scripted staging an optimizer would replace.

**Torso:** not locked. The waist is penalized, not constrained (`Waist Yaw` w 30,
leg-core). Your "assuming we do not use the torso joint, which we cannot guarantee
unless we lock it" is correct as stated — locking would need a model change or a
hard joint-range clamp.

**Note against the intuition:** the design spec (§7) says the **palm brace is
physically impossible on the real robot** — the magpie grippers occupy the palm, so
the legacy flat-hand plant was deleted and stage 34 goes straight to a forearm
brace. Your three-contact ladder (elbow → forearm → palm) therefore has a hardware
constraint on its third rung: bracing on the *side* of the rotated gripper (your
wrist-yaw idea) is the workaround, but it needs a collision proxy on the gripper,
which §9 lists as still missing (the gripper geom is `contype=0/conaffinity=0` —
literally phantom). **Any palm/gripper-brace result in sim today is not physical.**
That is the single most important thing to fix before trusting a three-contact
optimization.

## S1.3 The paper (26-0767, T-RO submission) — structure

**SCSP = CSO (selection) → CPO (planning)**, cascaded, both online.

- **CSO** answers *where to touch*. It is object-centric: it drops the robot's own
  contact geometry and asks, for candidate contact point `p_r` on the object surface
  and contact force `λ_r` in the friction cone, which point minimizes an object-motion
  cost (eq. 13–16). Three tricks make it tractable:
  1. **Discretize the surface** — offline approximate convex decomposition (CoACD)
     then farthest-point sampling → `n_s = 70` candidate points, each with a local
     frame `(n, t1, t2)` in a KD-tree; online "valid region selection" masks
     unreachable/occluded/unsuitable points (Alg. 1). This kills geometric
     nonsmoothness by replacing `min over ∂O` with `min over a finite set`.
  2. **Surrogate Contact Model (SCM)** — under *fixed contact set* (Assum. 1) and
     *multi-contact decoupling* (Assum. 2, block-diagonalize the Delassus matrix
     `W_env`), the environment force becomes an explicit piecewise-linear function of
     the robot force (Lemma 2, eq. 28). Complementarity disappears; error is bounded
     (Lemma 3).
  3. Result: a **MIQP** — binary = which candidate point, continuous = contact force —
     with a sub-ms inner QP (eq. 32).
- **CSO objective** (eq. 33–34) is worth stealing: not raw pose error, but
  `ℓ_stab` (distance to the nearest *statically stable* pose set) until you're in the
  goal's stable class, then `ℓ_pose`. Explicitly to stop the selector proposing
  contacts that are optimal-but-unstable for the downstream executor.
- **CPO** answers *how to get there*. Takes `p̂*_r` as a **prior, not a command** —
  a ranking strategy (eq. 37–40) compares the CSO optimum against the point nearest
  the current end-effector, with a normalized improvement ratio `ρ`, a threshold
  trigger `κ`, and a sliding-window/dwell filter `γ` (T1/T2/T3 = 5/10/15 steps) so
  the reference does not chatter. Then a lift/place potential-field cost (eq. 41–46)
  handles the gradient-sparse free-space approach.
- Solvers: OSC for the reduced end-effector problem; **sampling-based MPC (MPPI) for
  the full-order problem** — which is our regime.

## S1.4 Mapping SCSP → the H12 lean. What transfers.

| SCSP piece | Transfers? | For the lean |
|---|---|---|
| Cascade: global discrete selection → local continuous planner | **Yes, strongly** | This *is* your "offline strategy + online MPC refinement", with the useful twist that their selector runs online too |
| Discretize contact candidates, then MIQP | **Yes** | Our candidate set is tiny (elbow / forearm / gripper-side ≈ 3–20 points), so we skip CoACD+FPS entirely and hand-enumerate. The combinatorics that forced their machinery is *absent* for us |
| Friction-cone force optimization at selected points | **Yes** | This is exactly your "restorative bracing wrench" |
| `ℓ_stab` — prefer statically stable outcomes | **Yes, and it is the core of our objective, not a regularizer** | For us stability *is* the task: CoM/ICP inside the support polygon. Our `ℓ_cso` ≈ their `ℓ_stab` |
| Ranking + dwell filter to avoid contact chatter | **Yes** | Cheap insurance; partly already present as the 1.5 s new-pair ramp |
| SCM (block-diagonal Delassus, complementarity-free) | **Probably unnecessary** | Their reason for it is speed inside an MIQP over 70 points with a *moving* object. Our table is static and our contact count is ≤3 — a plain friction-cone QP per candidate set is affordable |
| Object-centric reduction: ignore robot-side contact location, optimize object-side | **No — must be inverted** | See below. This is the one real structural mismatch |
| Lift/place potential field for free-space approach | **Partly** | Our "approach" is the existing lean phase ladder, already tuned |

**The inversion.** SCSP's central simplification (eq. 13) is to *discard* the
robot-side contact point `{p_R,i}` and sample only the object surface `∂O`. For the
lean that is exactly backwards: the table is a flat plane with no interesting
structure, and the whole question — elbow vs forearm vs gripper — **is** the
robot-side point `p_R`. So we keep their algorithm and swap which manifold gets
sampled: **candidate points live on the robot's arm surface**, and the "object"
whose motion cost we minimize is the **robot's own floating base / CoM**.

That reframing is coherent and, I think, the single sentence that makes this idea
implementable: *the lean is SCSP with the robot's own body as the manipulated object
and the arm surface as the sampled contact manifold.* It also happens to be a fair,
substantive point for your review — the paper claims generality but its key reduction
is specific to robot-manipulates-external-object, and self-bracing/loco-manipulation
breaks it.

**Where our problem is harder than theirs**, and they will not help:
- Their "object" is a 0.1 kg rigid body with `M_o = diag(50,50,50,.05,.05,.05)`
  regularization. Ours is a 30-DOF underactuated floating-base humanoid whose base
  wrench is only realizable through the feet and the brace. Their eq. 15 "apply the
  best possible contact force" ignores whether the robot *can* generate it; for us
  actuator torque limits are the binding constraint and must be in the QP.
- They assume quasi-static and an impedance-controlled arm. The lean's failure mode
  is dynamic (capture point leaving the support polygon during the transition).

## S1.5 Assessment of the idea's feasibility

Encouraging, with one correction to the framing.

- **"It seems kind of like a linear programming problem"** — closer to a **QP or SOCP**
  than an LP, because the friction cone is second-order (linearizable to an LP with a
  polyhedral cone, exactly as the paper does in eq. 5 — so your LP instinct is
  recoverable, and that is the cheap first implementation). Adding *which* contacts
  are active makes it an **MIQP** / cardinality-constrained QP. With ≤3 candidate
  sites we can enumerate all 7 non-empty subsets exhaustively and skip integer
  programming entirely. That is a strong argument for doing this: **our combinatorial
  problem is small enough to solve by brute force**, which is precisely the thing the
  paper needed 20 pages of machinery to avoid.
- **"Back-compute where the lean needs to position the reaching arm"** — sound, and it
  is a standard static-equilibrium inverse problem, but it is *underdetermined*
  (redundant robot). It needs a regularizer, and the natural one is the paper's
  `ℓ_stab`: among all leans that reach the target, prefer the one whose CoM sits
  deepest inside the support polygon.
- **"If our optimization favored distributed load, triple-contact could emerge
  naturally"** — this falls out for free from a min-‖λ‖² objective over the friction
  cone: minimum-norm force distribution spreads load across available contacts.
  Emergent triple-contact is a *prediction of the formulation*, and a testable one.
  Good paper figure.
- **Main risk:** the static analysis will say a contact is sufficient, and the robot
  will still fall, because the loss of stability happens *during the transition*, not
  at the sustained pose. Mitigation is to keep the static solve as the *prior* (the
  paper's word) and let MPC own the transition — which is what you proposed anyway.

## S1.6 MEASURED: joint torques at the sustained brace

Tool added: `mjpc/lean_probe.cc` (+ CMake target `lean_probe`) in the worktree — a
headless driver mirroring `testspeed.cc`'s plan/step loop that logs
`actuator_force`, `ctrl`, CoM and summed robot↔`table_top_collision` force every
physics step. Run: `./build/bin/lean_probe --task "Lean H12" --strategy 4 --time 60
--threads 10`. Output `lean_analysis/lean_strat4.csv` (30 000 steps, dt 2 ms).

Phase timeline (auto-advanced, no manual scrubbing):
`stand_up` 0–5 s → `arm_extend_standing` 5–8 → `lean_with_arm_no_brace` 8–11 →
`arm_plant` 11–12.8 → `lean_forward` 12.8–20.8 → **`forearm_brace_lean` 20.8–60 (held)**.

Mean |τ| over the last 5 s (N·m): left_knee **128.6**, right_knee 31.6,
right_hip_roll 24.6, left_ankle_pitch 20.8, right_hip_pitch 18.8, left_ankle_roll 13.4,
… bracing-arm max = right_shoulder_pitch **4.6**, right_elbow 1.1, right_wrist_pitch 0.3.
Table contact force ≈ (−8, +12, **45**) N. Robot mass 68.07 kg → weight 668 N.

Share of summed |τ|: **legs 89 %, arms 9 %, torso 2 %.**

**Headline: the brace carries almost nothing.** 45 N normal ≈ 6.7 % of body weight,
while the left knee holds 129 N·m. This rollout is a lean that happens to touch the
table, not a load-sharing brace. Two consequences:

- It is a **baseline to beat**, not a target to reproduce. The restorative-wrench
  formulation should predict far more load through the arm; if it does and MPC can
  realize it, that difference is the paper's result.
- The contact-selection objective belongs on the **leg/support-polygon side**, not the
  arm side. Arm torques are ranks 14/19/22/24/25/26/27 of 27 — choosing elbow vs
  forearm vs palm barely moves arm torque directly; its leverage is what it *removes
  from the knee*.

Also found: the `forearm_brace_lean` contact pairs are bodies **25 = right_shoulder_yaw_link
(upper arm)** and **28 = right_wrist_pitch_link (wrist)** — they bracket the elbow rather
than name the forearm (`right_elbow_link` is body 26). Ask Allen whether that is deliberate.

Report artifact: <https://claude.ai/code/artifact/e828883b-db65-486f-8e7a-e0f5d52fb9e2>

Caveats: single seed, single run, not replicated.

### S1.6b No-brace baseline — the brace appears to buy nothing

`--strategy 2` (`h12_pipeline_lean_no_brace`, terminal phase `lean_with_arm_no_brace`),
same 60 s / same probe. Last 5 s:

| | braced (strat 4) | no-brace (strat 2) |
|---|---|---|
| left knee \|τ\| | **128.6** | **82.1** |
| right knee \|τ\| | 31.6 | 46.8 |
| Σ leg \|τ\| | 278.6 | 268.0 |
| Σ arm \|τ\| | 28.9 | 49.8 |
| CoM x | 0.788 | 0.791 |
| CoM z | 0.751 | 0.769 |
| mean cost | 354.2 | 226.5 |
| table contacts | 1.86 | 1.00 |

**The braced run reaches no further (CoM x 0.788 vs 0.791), loads the legs slightly
more (278.6 vs 268.0), puts 57 % more torque through the left knee, and costs 56 %
more.** On this evidence the hand-authored brace is not paying for itself. That is a
strong motivation for the whole project — but treat it as a flag, not a result yet,
because the comparison is confounded:

1. **The "no-brace" run is not contact-free** — it registers 1.0 table contact and
   ~48 N. Presumably the reaching hand rests on the table near the object (target
   x ≈ 1.4–1.5, table top z = 0.77). So this is *two-contacts vs one-contact*, not
   *brace vs no brace*.
2. **The table-force sign in `lean_probe.cc` is not trustworthy.** `mj_contactForce`
   returns the force in the contact frame acting on one geom of the pair; the probe
   sums frames without normalizing which side is the table, so signs flip with geom
   ordering (braced fz range −22.9…115.5, no-brace −64.7…0.0). Magnitudes are usable,
   directions are not.

**Fix before Session 2 concludes anything from this:** extend `lean_probe.cc` to log
per-contact `(geom1, geom2, body, force)` rather than a single sum, so each contact is
attributed to a specific link. Then re-run both. Also worth adding: CoP, and capture
point vs support polygon, since "did the lean get further" is the actual question and
CoM x alone is a weak proxy.

## S1.7 Session 2 is unchanged and now better specified

Build the static-equilibrium QP with: contact wrenches at 3 candidate sites, a
linearized (polyhedral) friction cone, actuator-torque limits, CoM-in-support-polygon
as the objective, min-norm force as the regularizer. Enumerate all 7 subsets. Then
map the reach-target grid → predicted contact set.

---

---

# SESSION 2 (2026-08-03, same day) — QP core works, pose stage does not

Files: `mjpc_icra2026/lean_analysis/contact_select.py`, `render_poses.py`.

## S2.1 What is validated

The **static-equilibrium QP itself is correct and checked against physics.**
Formulation, at a pose with qvel = qacc = 0:

```
base   (6 rows):  Σ_i J_i[:, :6]ᵀ λ_i  =  g[:6]           hard equality
joints (27 rows): τ = g[6:] − Σ_i J_i[:, 6:]ᵀ λ_i         |τ| ≤ τ_max
objective:        ‖τ / τ_max‖²  +  1e-4 ‖λ‖²
```

Contacts: 4 corners per foot (8) + one point per selected bracing site. Friction
is a linearized pyramid at μ = 0.6, applied by iterative clamping; normal forces
unilateral. Decision variables are the λ's; τ is recovered, not optimized directly.

Validation at the `stand_up` keyframe, legs only: base residual **0.00**, foot
normal forces sum to **663.6 N** against a robot weight of **667.7 N**, and the two
knees come out symmetric at **22.2 N·m**. That is the QP reproducing gravity
correctly, so the machinery is sound.

**Two real bugs found and fixed along the way:**

1. **`mj_inverse` is the wrong gravity term.** `qfrc_inverse` folds in
   `qfrc_constraint` — i.e. whatever contacts the pose happens to be penetrating.
   At the seed keyframe that was **20.2 kN** in the base-z row against a 668 N
   robot, giving base residuals of ~1e6. The correct term is `d.qfrc_bias` after
   `mj_forward` at rest. Worth remembering for any MJPC-side static analysis.
2. Base equilibrium was soft-weighted (1e3), which conflated "this contact set
   cannot balance the robot" with "this contact set exceeds torque limits". Now
   1e6, and the two failure modes are reported separately (`base_residual` vs
   `max_ratio`).

## S2.2 What is broken — the IK pose stage

**`solve_ik` returns converged-but-physically-impossible poses.** It has no
collision constraint and no posture regularization, so for a far target it solves
the reach by driving the torso straight *through* the table top. Rendered evidence:
`lean_analysis/poses_1.20_0.15_0.90.png` — all 8 subsets show the robot draped on
and through the table rather than leaning over it, with IK residuals of ~0.001
(i.e. "converged").

**Therefore every per-subset number produced so far is void.** No contact-set
ranking, no reach-target map. The docstring in `contact_select.py` now carries a
KNOWN BROKEN banner so the results are not picked up by accident.

This is the value of rendering before reporting: the residuals looked clean and the
torque ratios looked plausible (0.64 legs-only → 3.38 palm), and all of it was
nonsense.

## S2.3 Fix list for the next session, in order

1. **Collision in the IK.** Either run MuJoCo collision detection each Gauss-Newton
   iteration and add repulsion rows for any robot↔table penetration, or add explicit
   inequality rows keeping pelvis/torso/head above the table-top plane. The former
   is more general and the model already has the collision proxies.
2. **Posture regularization** toward the seed pose in the nullspace — currently
   absent despite the original docstring claiming it (now corrected). Without it the
   redundant DOFs wander into nonsense.
3. **Torso handling.** Decide explicitly whether `torso_joint` is locked (§S1.2: it
   is *not* locked today, only penalized). The formulation's "assume no torso" needs
   to be either enforced or dropped.
4. **Sane target range.** Table top z = **0.865**, x from **0.298 to 1.522**; the
   `stand_up` base sits at x = 0.19, right at the front edge. A target at x = 1.2
   demands leaning across most of the table. Sweep should start near the front edge
   and walk outward so the infeasibility boundary is found, not jumped over.
5. Only then: enumerate the 8 subsets across a target grid and build the map.

## S2.4 One number that does survive, and matters

Actuator limits (from the model, not assumed): knee **300 N·m**, hip pitch/roll 300,
hip yaw 200, torso 200, ankle 75, shoulder pitch/roll **32**, shoulder yaw/elbow
**14.4**, wrist **9.5**.

So the measured 128.6 N·m left knee from Session 1 is **43 % of limit, not
"near-saturating"** — S1.6's wording was wrong and is corrected in the artifact.

More importantly: **the bracing arm's torque ceiling is tiny.** An arm whose largest
joint limit is 32 N·m cannot push very hard on a table regardless of what the
selector asks for. This is very likely the binding constraint in the whole
formulation, and it is a constraint the paper's CSO does not model at all (its eq. 15
assumes the best possible contact force is available). **Predicted outcome: contact
selection for the lean is limited by arm actuator torque, not by friction or by
geometry** — which, if it holds up, is both the reason distributed multi-contact
should win and a genuine point of difference from SCSP worth writing up.

---

# SESSION 3 (2026-08-03) — IK still not converging; root cause identified

Worked the §S2.3 fix list. Items 1–3 implemented, items 4–5 still blocked.

## S3.1 Implemented

- **Non-penetration in the IK** (`collision_rows`): one row per penetrating contact
  along the contact normal, taken from MuJoCo's own narrowphase after `mj_forward`,
  so it sees exactly the collision proxies the planner sees. Robot bodies are
  **never** exempted — an earlier version exempted the feet and the selected site's
  body, which let the whole upper-arm link sink 47 mm through the tabletop. A 2 mm
  `PEN_TOL` lets a resting contact sit at dist ≈ 0 without fighting the site task.
- **Posture regularization** rows toward the seed pose (`W_POST`).
- **`lock_torso` flag** on `solve_ik`. Documented answer to §S2.3 item 3: the shipped
  model does **not** lock `torso_joint` — it is only penalized (`Waist Yaw` w 30 in
  the leg-core). `lock_torso=False` is therefore the faithful default; `True` freezes
  it at the seed value and is the "assume we do not use the torso" case. Both are
  now available to compare rather than assumed.
- **Per-task residuals** instead of one lumped norm — `reach`, `foot`, `site_max`,
  `penetration` — so "the hand missed the target" is distinguishable from "the elbow
  cannot reach the table". This was necessary; the lumped norm hid everything.

## S3.2 A scenery artifact that wasted a cycle

Every subset reported a constant 42.5 mm penetration. It is the **object penetrating
the table** at the seed keyframe — scenery vs scenery, nothing to do with the robot.
It was correctly excluded from the constraint rows but wrongly included in the
reported depth. Fixed. Flag for Allen: the `stand_up` keyframe has the manipulation
object embedded 42.5 mm into the tabletop.

## S3.3 Still broken, and now precisely diagnosed

`solve_ik` **does not converge**. Task residuals stall with the **feet flying off
their pins by 0.2–1.7 m**, which is physically impossible and makes every downstream
number void. Three hypotheses were tested and two were refuted:

| Hypothesis | Test | Result |
|---|---|---|
| Foot pin outweighed by collision rows | raise `W_FOOT` 10 → 300 (30×) | **Refuted.** foot residual 0.283 → 0.294, i.e. no effect |
| Inequality-as-equality rows chatter | chain 30 × 20-iteration warm-started calls | **Refuted, and the evidence was bad** — that path re-captures `foot_targets` each call, so the feet drift and the 0.000 residual is measured against a moving reference |
| Posture reg fights the deep lean | sweep `W_POST` 0.6 / 0.05 / 0.01 | **Refuted, and inverted** — *lowering* it made things far worse (foot 0.294 → 1.667) |

The weight-insensitivity across a 30× change is the tell: this is not a trade-off
being lost, it is a solver that is not descending. Root cause:

1. It forms **weighted normal equations** `A'A + 1e-4·I` with row weights up to 300,
   so conditioning is ~1e5 and the 1e-4 damping is meaningless.
2. It then **clips the step per-component** at 0.08, which for a badly scaled system
   destroys the descent direction — 800 such steps are a random walk.
3. Non-penetration is an **inequality** stacked as an equality row that switches on
   and off between iterations.

**Fix (next session, item 1):** solve each Gauss-Newton step as a real QP —
feet as hard equality, non-penetration as inequality, posture in the objective —
using `quadprog` (already installed; `scipy` has no good QP). Weight tuning is not
the answer and should not be attempted again.

The KNOWN BROKEN banner in `contact_select.py` is updated with this diagnosis.
The QP core from §S2.1 remains validated and untouched.

## S3.4 Honest status

Two sessions in, the deliverable (reach-target map coloured by contact set) does not
exist, and no contact-set ranking has been produced. What exists is: a validated
static-equilibrium QP, a correct set of actuator limits, a measured baseline of the
current hand-authored brace, and an IK whose failure is now understood well enough to
fix in one focused change rather than by tuning.

---

# SESSION 4 (2026-08-03) — IK FIXED. Poses are physical.

## S4.1 The fix

`solve_ik`'s Gauss-Newton step is now a real QP per iteration (`quadprog`):

```
min ½ dq'G dq − a'dq
s.t.  n'J_c dq ≥ depth_c − PEN_TOL      non-penetration, INEQUALITY
      −TRUST ≤ dq_i ≤ TRUST             trust region, INSIDE the QP
```
with reach, bracing sites, feet and posture as weighted objective terms plus a
Levenberg term to keep `G` positive definite. This replaced the weighted normal
equations and the post-hoc step clipping.

Result at three targets × several subsets: **reach error ~4–9 mm, foot error
0.0000 m, penetration exactly at the 2 mm tolerance, zero QP fallbacks.** For
comparison, the old solver stalled at foot errors of 0.2–1.7 m. Weight tuning was
correctly ruled out in §S3 — the solver was the problem.

Three further bugs found and fixed on the way, in order of how much they mattered:

1. **Non-penetration had the wrong sign.** It read `n'J dq ≥ −depth`, which
   *permits* sinking further by up to the current depth. Correct form is
   `n'J dq ≥ depth − PEN_TOL`.
2. **Hard foot equality + collision inequalities is usually infeasible** inside a
   trust region, and the infeasibility handler was silently dropping
   non-penetration — measured **245 of 250 iterations**. Feet moved into the
   objective at `W_FOOT = 300`; they hold to <1e-3 m in practice, and
   non-penetration stays hard, which is the thing that must not be compromised.
3. **Collision was reactive, not predictive.** MuJoCo only reports a contact once
   geoms overlap, so the IK learned about the table only from inside it, and
   climbing out within one trust region was infeasible. Fixed by inflating
   `geom_margin` to `IK_MARGIN = 0.025` for the IK model instance, so near-contacts
   are constrained *before* they overlap. This is what took fallbacks to zero.

Also: a bracing site now counts as **placed** when its link is genuinely in contact
with the table, read from MuJoCo's contacts — not when the nominal body-frame point
reaches the table plane. The point sits inside the link, so a resting contact still
leaves ~35 mm of link thickness; the old criterion called correct poses failures.

## S4.2 Renders — poses are physical now

`poses_0.95_0.15_0.90.png`: all 8 subsets show the robot leaning over the table,
torso above the surface, arms resting on the top, feet planted behind. Compare
`poses_1.20_0.15_0.90.png` from §S2.2, where the body was inside the tabletop.

## S4.3 First trustworthy signal

At reach target (0.95, 0.15, 0.90), max |τ|/limit by subset:

| subset | max ratio | brace force (N) | placed |
|---|---|---|---|
| legs only | 0.62 | 0 | — (base residual 5.03 → cannot balance) |
| elbow | 0.81 | 130.2 | **no** — upper arm cannot reach the table alone |
| forearm | 0.42 | 127.2 | yes |
| palm | 0.65 | 58.0 | yes |
| elbow + forearm | 0.49 | 126.9 | yes |
| elbow + palm | 0.43 | 107.4 | yes |
| forearm + palm | 0.48 | 105.5 | yes |
| **elbow + forearm + palm** | **0.40** | 112.0 | yes |

Two things worth noting, both provisional until the sweep lands:

- **Triple contact minimises peak actuator load** (0.40), and adding contacts
  monotonically reduces it from the legs-only 0.62. That is the predicted
  emergence-of-distributed-load result, appearing from a min-effort objective
  without being asked for.
- **Elbow-alone is not placeable.** The upper arm cannot reach the table without the
  hand contacting first — geometrically the distal links get there first. So the
  "elbow-only brace driven by shoulder roll+pitch" rung of the intuition does not
  exist as an independent option on this robot without rotating the arm out of the
  way. Worth checking against Allen's image, which appeared to show exactly that.

## S4.4 Running

`sweep.py` over x ∈ [0.40, 1.35] × z ∈ [0.88, 1.16] at y = 0.15, all 8 subsets
per target (~320 IK+QP solves), selecting by min normalized effort with ties broken
toward fewer contacts. Output `sweep.json` → reach-target map.

---

# SESSION 5 (2026-08-03) — THE MAP EXISTS

Sweep done: 45 targets (x ∈ [0.40, 1.36] step 0.12, z ∈ [0.88, 1.16] step 0.07,
y = 0.15) × 8 contact sets = 360 IK+QP solves. `sweep.json`, `map.json`.
Map artifact: <https://claude.ai/code/artifact/e2fa59b0-b081-4f91-97f2-8266cb86f4e7>

## S5.1 The result

Selected set (min normalized effort among admissible), by reach target:

```
z=1.16 |  fa+palm  fa+palm     palm     palm  forearm  forearm  forearm  forearm  forearm
z=1.09 |  fa+palm     palm     legs     palm  forearm  forearm  forearm  forearm  forearm
z=1.02 |  fa+palm     palm     palm  forearm  forearm  forearm  forearm  forearm  forearm
z=0.95 |     palm     palm     palm  forearm  forearm  forearm  forearm  forearm  forearm
z=0.88 |     palm     palm     palm  forearm  forearm  forearm  forearm  forearm     ---
          x=0.40    0.52     0.64     0.76     0.88     1.00     1.12     1.24    1.36
```

**Near reach → gripper palm. Far reach → forearm.** Clean and monotone, with the
switch at x ≈ 0.70 m. This is the user's predicted structure — "a nearby reach needs
only a small restorative wrench, most easily satisfied by the gripper palm brace;
a far reach wants the proximal brace" — recovered from a min-effort objective without
being encoded. Palm braces really are light: **2.6 N** at the nearest target,
5.1 N at (0.52, 0.95).

The one substitution: the far-field pick is the **forearm**, not the elbow, because
the elbow is not placeable (below).

## S5.2 The three claims, tested

**1. Triple contact minimising load — NO, and §S4.3 was over-read.** Across 45
targets `elbow+forearm+palm` is the min-**effort** choice **0/45**. It is the
min-**peak-torque** choice **10/45**. §S4.3 looked at one target, on `max_ratio`,
and generalized it. Peak torque and total effort disagree, so *which objective* is
the modelling choice that decides whether distributed multi-contact emerges at all.
That is worth stating explicitly in the paper rather than picking one silently.

**2. Elbow alone — essentially never placeable, 2/45.** The upper arm cannot reach
the table without the distal links touching first. `elbow+palm` is also 2/45 —
skipping the intervening forearm is geometrically awkward. So the elbow rung of the
1:1 contact↔actuator ladder does not exist as an independent option under the current
site definitions; it would need the arm rotated to clear the hand. **Check this
against Allen's image**, which appeared to show exactly an elbow brace.

**3. Legs-only boundary — bracing strictly required at 23/45.** The boundary is
diagonal: at z = 0.88 the legs cope out to x = 0.76; at z = 1.16 they cope out to
x = 1.00. **Reaching low costs stability, reaching high does not** — which supports
the "near but low-Z still needs extra contacts" half of the intuition.

Admissibility across the grid: forearm 44/45, forearm+palm 44/45, triple 43/45,
palm 38/45, elbow+forearm 34/45, legs 22/45, elbow+palm 2/45, elbow 2/45.

## S5.3 What bounds this

- **Selected ≠ required.** At many near targets the legs alone are admissible and the
  palm wins on effort by a hair (0.18 vs 0.21 at one checked target).
- **Static only.** Says nothing about whether the *transition* into the pose is
  stable, which is where the real lean fails. This is a prior for the planner, not a
  controller — the paper's own framing of CSO output as prior, not command.
- **One deterministic solve per cell.** No seeds, no replicates.
- **Torso not locked** (faithful to the shipped model). `sweep.py --lock-torso` is
  implemented but not yet run — that is the user's "assume no torso" case and is the
  next comparison.
- **The gripper collision proxy is still phantom** (§S1.2). Every "palm" result here
  relies on the wrist-yaw capsule/sphere proxies, not a real gripper. Since palm wins
  the entire near field, this is now the highest-value model fix in the project.

---

# SESSION 6 (2026-08-03) — §S5.2 claim 2 was MY BUG. Map redone.

## S6.1 The elbow site was defined in empty space

`SITES['elbow']` was `right_shoulder_yaw_link` + `[0.002, 0.007, −0.250]`, taken from
the collision geom's `geom_size`. **For a MESH geom, `geom_size` is the bounding-box
half-extent, not the shape.** The upper-arm body origin is at world z = 1.308 and the
elbow joint anchor at 1.126 (body-frame z = **−0.182**), but the bounding box extends
to −0.256 — so my point sat ~68 mm *past* the joint, floating outside the robot.

Placeability of an elbow-only brace, 12 probe targets:

| elbow site (body-frame z) | placed |
|---|---|
| −0.250 (old, from `geom_size`) | **1 / 12** |
| −0.182 (at the elbow joint) | **12 / 12** |
| −0.150 (mid upper arm) | 12 / 12 |
| proximal end of forearm link | 12 / 12 |

So §S5.2's "the elbow is essentially never placeable, 2/45" was an artifact of my own
site definition, not a property of the robot. Corrected to −0.182 and the whole sweep
re-run; `sweep_BADELBOW.json` / `map_BADELBOW.json` retain the void results.

**Allen's photograph is consistent after all** — an elbow brace is reachable at every
target tested. The earlier note asking him to explain it can be dropped.

## S6.2 Corrected map

```
z=1.16 |  eb+palm   eb+fa     palm   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa
z=1.09 |  eb+palm   eb+fa    eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa
z=1.02 |    eb+fa    palm    eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa
z=0.95 |     palm    palm    eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa
z=0.88 |     palm    palm    eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa   eb+fa
          x=0.40    0.52     0.64    0.76    0.88    1.00    1.12    1.24    1.36
```

**Near + low → palm alone (6/45). Everything else → elbow + forearm (37/45)**, i.e. the
whole upper limb laid along the table. `elbow+palm` picks up the remaining 2.

This is a *better* match to the original intuition than the broken version was: the
near/small-wrench → palm rung survives, and the far rung is an elbow brace (with the
forearm alongside) rather than the forearm alone.

Admissibility: elbow, elbow+forearm, elbow+palm, triple all **45/45**; forearm and
forearm+palm 44/45; palm 38/45; legs-only **22/45** (unchanged — legs-only never
involved the elbow site, which is a useful consistency check that the re-run is sane).

## S6.3 Min-effort vs min-peak-torque — recommendation

They genuinely disagree:

| | min-effort Σ(τ/τmax)² | min-peak max(τ/τmax) |
|---|---|---|
| elbow+forearm | **37** | 19 |
| elbow+forearm+palm | 0 | **16** |
| palm | 6 | 5 |
| elbow | 0 | 3 |

Triple contact is *never* the cheapest but is the least-saturated choice a third of the
time. **Recommendation: report both, and lead with min-peak-τ.** Min-effort answers
"cheapest to hold"; min-peak answers "furthest from saturation". For an arm whose
largest joint limit is 32 N·m (§S2.4), distance-to-saturation is what predicts what
survives contact with the real robot — and it is also the objective under which the
distributed-load / triple-contact story is true rather than wishful.

## S6.4 Process note

The first relaunch of the corrected sweep was silently killed: `nohup … &` inside a
backgrounded tool call dies when the wrapper shell exits, and the completion
notification refers to the wrapper, not the job. Only 14/45 targets ran. Launch long
jobs as a plain foreground command inside a backgrounded call instead.

## S6.5 Locked torso — structure survives, margin does not

`sweep_locktorso.json` vs `sweep.json`, same grid, torso frozen at its seed value.

- **Selection barely changes: 6 / 45 cells differ.** Near+low still picks palm,
  everything else still picks elbow+forearm. Legs-only admissibility is **identical
  (22/45)**, which also serves as a consistency check on the two runs.
- **But the margin cost is real.** Averaged over the grid, the best available option
  under a locked torso costs **+20.5 % effort** and **+9.9 % peak torque**; locking is
  worse at 29/45 cells and better at 16. Most importantly the **worst-case peak torque
  across the grid goes from 0.42 to 0.89 of limit** — from comfortable to nearly
  saturated.

**Answer to the user's question** ("assuming we do not use the torso joint, which we
cannot guarantee unless we lock it"): the assumption is **safe for deriving the contact
selection** — the map is essentially the same shape — but **not safe for sizing the
result**, because the worst-case actuator margin more than doubles its utilisation.
So: derive the strategy with the torso locked if that simplifies the story, then
re-check torque margins with it free before claiming anything about feasibility on
hardware.

---

# SESSION 7 (2026-08-03) — the gripper is NOT phantom, and it changes the answer

## S7.1 Correction: I was quoting a stale design doc, not the model

Since §S1.2 I have repeatedly warned that "the magpie gripper geom is
`contype=0/conaffinity=0` — a phantom with mass but no collision, so any palm-brace
result in sim is not physical." **That is no longer true and I never checked the
model.** It comes from `docs/specs/2026-06-22-lean-pipeline-31-35-design.md` §9, which
is a *plan*, and the branch has since implemented it.

`Lean_H12_Magpie.xml` actually carries, per gripper:

| geom | type | contype / conaffinity |
|---|---|---|
| `right_gripper_collision` | box, half-extent (0.042, 0.032, 0.067) @ x=0.096 | **1 / 1** |
| `right_gripper_jaw_a` | box @ (0.18, −0.004, −0.08) | **1 / 1** |
| `right_gripper_jaw_b` | box @ (0.18, −0.004, +0.08) | **1 / 1** |
| `right_gripper_keepaway` | box @ x=0.10 | 2 / 2 (separate layer) |
| `right_wrist_pad` | capsule | 0 / 0 (visual) |

So the proxy work in spec §9 is **done**. Lesson: read the model, not the plan doc.
No action needed from Allen on this.

## S7.2 The real issue is different, and worse for the headline

Everything through §S6 ran on **`Lean_H12.xml` — the handless model**. Its "palm" is a
wrist capsule + a 33 mm sphere at x = 0.13. That is a bare wrist stub, not a gripper.
The hardware configuration is the Magpie model, where a 0.506 kg gripper with jaws
extends to x ≈ 0.226.

Spot check at two targets (min-effort, torso free):

| target | subset | base model | Magpie model |
|---|---|---|---|
| (0.52, 0.95) | palm | placed, ratio 0.18, **eff 0.064** | **NOT PLACEABLE**, eff 0.289 |
| (0.52, 0.95) | elbow+forearm | placed, eff 0.150 | placed, **eff 0.254** |
| (1.00, 0.95) | palm | placed, ratio 0.69, eff 1.189 | placed, ratio 0.77, eff 1.566 |
| (1.00, 0.95) | elbow+forearm | placed, eff 0.243 | placed, eff 0.334 |

**With the real gripper fitted, the palm brace stops being placeable at the near
target** — the gripper body/jaws reach the table before the nominal palm point — and
`elbow+forearm` becomes the cheapest option there too. If this holds across the grid,
the "near + low → palm" half of §S6.2 is an artifact of analysing the handless robot,
and the answer collapses toward elbow+forearm nearly everywhere.

Note this does not vindicate the old phantom-gripper warning: the gripper is *more*
present than I assumed, and that is precisely why palm loses.

`contact_select.py` now takes `LEAN_MODEL=magpie` to select the variant. Base sweeps
preserved as `sweep_base_free.json` / `sweep_base_locktorso.json`; hardware run is
`sweep_magpie.json`.

## S7.3 Magpie sweep result — the palm result does not survive

| | handless (`Lean_H12`) | **grippers fitted (`Lean_H12_Magpie`)** |
|---|---|---|
| palm placeable | 45/45 | **17/45** |
| palm admissible | 38/45 | **4/45** |
| palm min-effort wins | 6 | **0** |
| elbow+forearm wins | 37 | **27** |
| legs-only admissible | 22/45 | **14/45** (bracing required 31/45) |
| no feasible set | 1 | **7** |

**"Near + low → palm" was an artifact of the handless model.** With the grippers on,
the palm never wins and is barely ever admissible: the 0.5 kg gripper body and jaws
reach the table before the palm reference point. `elbow+forearm` — the whole upper limb
along the table, which is what Allen's photo shows — dominates at 27/45.

The distributed-load story gets *stronger* on hardware: triple contact is the
min-peak-torque choice at **20 of 38 solved targets** (handless: 16/45), while still
almost never the min-effort choice (3/45). Same objective split as §S6.3, sharper.

Extra 1 kg at the ends of the arms also measurably shrinks the unaided envelope:
legs-only 22/45 → 14/45.

## S7.4 The lowest row is unsolved, not impossible — and it is the user's own point

Seven targets have **no feasible contact set**, all at z = 0.88 (15 mm above the
tabletop). Cause: the gripper jaws extend **±106 mm** from the grasp axis, so the grasp
point cannot sit 15 mm above the table with the jaw axis vertical — the lower jaw would
be ~91 mm inside it.

This is exactly the user's "we need to rotate the gripper 90° to brace against its
side". Low targets *require* a wrist roll putting the jaw axis horizontal.
`wrist_roll` (±2.967 rad) is available to the IK but there is **no term rewarding it**,
so the solver never finds that branch. **Report the z = 0.88 row as "not solved", not
"infeasible"**, until a gripper-orientation task is added. That is the top of the next
session's list, and it converts a modelling gap into a testable prediction of the
user's own intuition.

## S7.5 Not done this session

Planning-rate check (`lean_probe` on `Lean H12` vs `Lean H12 Magpie`) and the MJPC
handoff sketch (§S1.2 `contacts` array + `brace_force_target`) were both deferred —
the model correction and re-sweep took the session.

---

# BRANCH SITUATION + NEXT-SESSION DIRECTION (user note, 2026-08-03 late)

## B.1 We are already on `asset-unification` — provenance label was wrong

The worktree `/home/humanoid/Programs/mjpc_icra2026` was switched **icra2026 →
asset-unification at 13:34** today (reflog), and now carries:

- `5691654` assets: make CL_Assets the source of truth for the H1-2 robot (17:35)
- `ca336fc` tasks: clamp the jab keyframes to the serviced elbow limit (17:35)
- `8a452e2` avoid: declare the sensor ranges the residual always reads (17:52)

Almost certainly the concurrent session (see memory `concurrent-claude-sessions-mujoco`).
**User's request to stay off `icra2026` is already satisfied.**

**Correction to earlier sections:** §S4–§S7 say "branch icra2026". They actually ran on
`asset-unification` — the built `Lean_H12_Magpie.xml` was regenerated at 17:54 and every
sweep ran after that. Checked whether it matters: the *only* lean-model diff between the
branches is two `jab_*` keyframes (elbow −1.00 → −0.95), irrelevant here. **Results stand;
only the label was wrong.** Sessions 1–2 (`lean_probe` torque data, ~12:00–12:29) genuinely
were on icra2026, pre-CL_Assets.

## B.2 What the user asked for next

> "let's try using the spherical collision geom from CL_assets, and the visual mesh as
> collision mesh, just to see how they do. i'm worried about interfering with allen on
> icra2026, so let's keep on asset-unification"

Two collision-geometry variants to try for the gripper / arm links, and compare against
the current box proxies:

1. **Spherical collision geom from CL_Assets.** Candidate source trees:
   `/home/humanoid/HAMS-grasp/CL_Assets` and
   `/home/humanoid/Programs/Humanoid_Simulation/CL_Assets`. Find the sphere proxy the
   H1-2 assets ship and wire it in.
2. **Visual mesh as collision mesh.** Note the design spec §9 explicitly warns against
   this (real-time planner cost + spurious self-contacts) — so this is a deliberate
   "just to see how they do" experiment, and the **planning-rate check becomes
   mandatory**, not optional, for this variant.

Why it matters here: §S7.3/§S7.4 showed the palm result is entirely a function of gripper
collision geometry — palm went from winning 6/45 (handless) to admissible 4/45 (box
proxies). A sphere proxy will move that number again, and the honest question is which
geometry is the right one to believe. Run all three (handless / box / sphere / mesh) over
the same grid and report the palm and elbow+forearm shares side by side.

Carry-over, still not done: gripper-orientation task for the unsolved z=0.88 row (§S7.4),
planning-rate check, MJPC handoff sketch (§S1.2).

---

# SESSION 8 (2026-08-04) — collision-geometry comparison: it's the JAWS

Branch confirmed `asset-unification` @ `8a452e2`. No other jobs running at start.

## S8.1 There is no spherical *collision* geom in CL_Assets

The two `type="sphere"` geoms in `CL_Assets/mujoco_assets/h1_2_magpie.xml`
(lines 266, 318) are `contype=0 conaffinity=0 group=3`, alpha 0.35 — **visual
markers**. What CL_Assets actually ships as the wrist proxy is a **cylinder**
(`h12_collision`, r = 32 mm, half-length 12.5 mm), and its gripper
(`magpie_gripper.xml`) uses **full mesh collision on every part** (mount, base,
cranks, fingers, rockers) with articulated jaw joints.

Built a 32 mm **sphere** proxy matching the cylinder radius as the nearest thing to
what was asked. Ask the user whether they would rather have the cylinder verbatim.

## S8.2 Variants built (all inside `build/`, source tree untouched)

`build/mjpc/tasks/humanoid_bench/h1_2/h1_2_modified_magpie_{sphere,mesh}.xml` plus
`lean/Lean_H12_Magpie_{sphere,mesh}.xml`. `contact_select.py` now takes
`LEAN_MODEL=handless|magpie|sphere|mesh`.

**Caveat on the "mesh" variant:** the MJPC magpie model represents each gripper as a
*single rigid body* carrying a `h12_mount` mesh + three hand-authored boxes. There is
no full-gripper mesh in this model to promote — so "visual mesh as collision mesh"
here means **the mount mesh only** (bounding radius 46 mm), which is *less* geometry
than the box proxies, not more. A true full-mesh test needs the CL_Assets gripper
hierarchy imported, which brings extra DOFs. Do not read this row as "CL_Assets
full-mesh gripper".

## S8.3 Four-way result

| variant | palm placeable | palm admissible | palm wins | elbow+forearm wins | triple wins | legs-only | no feasible set |
|---|---|---|---|---|---|---|---|
| handless | 45/45 | 38 | 6 | **37** | 0 | 22 | 0 |
| **box proxies + jaws** | **17/45** | **4** | **0** | **27** | 3 | 14 | **7** |
| sphere (no jaws) | 45/45 | 39 | 6 | 8 | **29** | 20 | 0 |
| mount mesh (no jaws) | 45/45 | 39 | 6 | 8 | **29** | 20 | 0 |

**The decisive variable is whether the JAWS are modelled — not box vs sphere vs
mesh.** The box variant is the only one that represents the jaws (they reach
+/-80 mm from the gripper axis, +/-106 mm including thickness). Every variant without
jaw geometry restores the palm and flips the dominant selection to **triple contact
(29/45)**; the one with jaws kills the palm entirely and produces 7 unsolvable
targets.

Sphere and mount-mesh give *identical aggregate counts* but **not identical
solutions** — 39 numeric fields differ, max |delta| 0.395. The coincidence is at the
level of which subset wins, which is a coarse statistic. Do not report them as the
same result.

**So the modelling decision that matters is: are the jaws load-bearing geometry or
not?** Physically the jaws are real and would hit the table, so the box variant is
the honest one — which means S7.3's pessimistic palm result stands, and the
optimistic sphere/mesh numbers are an artifact of deleting geometry that exists.

## S8.4 Actuator-limit discrepancy — unresolved, and it undercuts S2.4

| joint | MJPC lean model | CL_Assets `h1_2_magpie.xml` |
|---|---|---|
| shoulder pitch/roll | +/-32 | +/-40 (`shoulder1`) |
| shoulder yaw / elbow | +/-14.4 | +/-18 (`shoulder2` / `elbow`) |
| **wrist** | **+/-9.5** | **+/-19** |

The wrist differs by **2x**. S2.4's claim that "the bracing arm's torque ceiling is
the binding constraint on the whole formulation" rests on the MJPC numbers. Since
`asset-unification` is explicitly making CL_Assets the source of truth for this robot,
**the lean model's arm limits may be stale**. Resolve before that argument goes in a
paper; re-run selection under CL_Assets limits to see whether the conclusion survives.

## S8.5 Not verified / not done

- **Renders were produced but could not be viewed this session** — the image-read tool
  failed repeatedly with a hook timeout, as did the Edit tool (this section was
  appended via Bash). `poses_0.52_0.15_0.95.png` (sphere variant) exists on disk,
  unviewed. The sphere/mesh rows above are therefore **numerically computed but not
  visually checked**, contrary to the standing rule. Check them first next session.
- Planning-rate check (`lean_probe`, mandatory for the mesh variant per design spec 9)
  still not run.
- Gripper-orientation task for the z = 0.88 row; MJPC handoff sketch.

---

# SESSION 9 (2026-08-04) — the contact set in the QP is NOT the contact set in the pose

Image reading was still blocked by hook timeouts, so instead of eyeballing renders I
wrote `lean_analysis/pose_audit.py`, a numeric substitute that checks the failure
modes a filmstrip catches: body clearance over the tabletop, feet planted, joints
pinned at their limits, and — the decisive one — **which bodies are actually
touching the table**. It caught something a render might well have hidden.

## S9.1 92 % of solved poses carry unmodelled contacts

Sampled 8 reach targets x 8 contact subsets = 64 poses (sphere variant):

```
poses with table contacts NOT in the modelled set:  59 / 64   (92 %)

left_hip_pitch_link    51      torso_link              39
right_hip_pitch_link   51      right_wrist_* chain     23-26
right_hip_yaw_link     45      right_elbow_link        16
left_hip_yaw_link      45      right_shoulder_yaw_link 14
```

**The hips are resting against the table edge in ~80 % of poses and the torso in
~60 %.** The equilibrium QP only ever includes the two feet plus the selected arm
sites — so it has been solving a contact set that **does not match the pose it was
handed**. Whatever load the hips and torso carry against the table edge is invisible
to the QP, which then attributes the whole restorative wrench to the feet and the
chosen arm contacts.

Cause is geometric and unavoidable given the setup: the table front edge is at
x = 0.298, the robot base is pinned at x = 0.19, and the tabletop is at 0.865 — i.e.
**hip height**. Leaning forward presses the pelvis into the table edge. A human
leaning over a table does exactly this.

## S9.2 What this invalidates, and what survives

**Invalidated:** the absolute force/torque numbers per contact set. The QP's
`brace_force`, `max_ratio` and `effort` are computed for a fictional contact set and
will over-attribute load to the arm.

**Partially survives:** the *comparison* between contact sets, because every variant
shares the same unmodelled hip/torso support — so the ranking is between
"elbow+forearm + hips/torso" and "palm + hips/torso". Directionally informative,
quantitatively not trustworthy.

**Unaffected:** S2.1's validation (that was at the standing keyframe with no table
contact), the actuator limits, and the S8.3 finding that jaw geometry is the decisive
modelling choice (that one is about placeability, not forces).

## S9.3 This is also a genuine result, not only a bug

The hip/torso-against-the-table-edge contact is a **real bracing surface that the
3-site ladder omits**. A human leaning over a table braces on their hips first and
their hands second. If the optimizer is allowed to use it, "which contacts does the
lean need" may have a different and better answer than any elbow/forearm/palm
combination — and it costs no arm torque at all, which matters given the arm's
32 N.m ceiling (S2.4).

**Recommended next step:** add `hip` and `torso` as candidate contact sites alongside
elbow/forearm/palm, re-run the enumeration over the enlarged candidate set (2^5 = 32
subsets, still trivially enumerable), and re-check the map. This directly tests
whether the interesting answer was outside the hypothesis space all along.

## S9.4 Method note

`pose_audit.py`'s clearance check is x-naive: it flags hip links as "below the
tabletop" when they are in fact *in front of* the table edge, not under it. Harmless
here (the touching-body list is what mattered) but fix before reusing it.

**Verification of the S8.3 sphere/mesh rows is still not done visually** — two
sessions running. The numeric audit is arguably stronger, but the renders on disk
remain unviewed.

## S9.5 Not done

Actuator-limit resolution (S8.4), planning-rate check, gripper-orientation task,
MJPC handoff sketch. Session consumed by the audit and its consequences.

---

# SESSION 10 (2026-08-04) — trunk contacts belong in the answer; arm torque limits are STALE

## S10.1 Adding hip + torso: they are in the optimal set almost everywhere

Added two trunk candidate sites, both anchored on PRIMITIVE geoms of `torso_link`
where `size` is the true shape (so the S6.1 mesh-bounding-box trap does not apply):

- `hip`   = front surface of the `hip` capsule (local `[0.05, 0, -0.05]`, r = 0.05)
- `torso` = lower front edge of the `torso` box (local `[0.07, 0, 0.05]`, hx = 0.07)

Enumerated all 2^5 = 32 subsets over a coarser 8-target grid (256 IK+QP solves):

```
x=0.52 z=0.95 -> palm
x=0.76 z=0.95 -> elbow+forearm+torso
x=1.00 z=0.95 -> elbow+forearm+hip+torso
x=1.24 z=0.95 -> elbow+forearm+hip
x=0.52 z=1.09 -> elbow+forearm+hip+torso
x=0.76 z=1.09 -> elbow+forearm+palm+hip+torso
x=1.00 z=1.09 -> elbow+forearm+hip
x=1.24 z=1.09 -> elbow+forearm+hip
```

**7 of 8 targets select a set containing a trunk contact.** `elbow+forearm+hip` and
`elbow+forearm+hip+torso` dominate. No target selects a trunk-ONLY set, so the arm is
still needed — the trunk supplements rather than replaces it.

This confirms S9.3: the hypothesis space was too small. The 3-site elbow/forearm/palm
ladder omitted a contact the optimizer wants at nearly every target, and one that
costs **zero arm torque**. Any paper claim of the form "the optimizer selects X" must
be made over the enlarged candidate set, or it is selecting the best option from an
arbitrarily truncated menu.

## S10.2 RESOLVED: the lean model's arm torque limits are stale, and CL_Assets wins

`mjpc/tasks/humanoid_bench/h1_2_base/_gen_h12_base_limits.py` (new on
`asset-unification`, 2026-08-03) states it outright:

> CL_Assets IS THE SOURCE OF TRUTH for the robot (2026-08-03). Its
> `mujoco_assets/h1_2_handless.xml` matches `h12_safety_layer/core/joint_limits.py`
> exactly on all 27 motorised joints -- both URDF_POSITION_LIMITS (joint range) and
> URDF_TORQUE_LIMITS (actuatorfrcrange). [...] arm actuatorfrcrange 120/120/75/25 vs
> the real 40/18/18/19

Verified directly in `CL_Assets/mujoco_assets/h1_2_handless.xml`: elbow
`actuatorfrcrange="-18 18"`, all three wrist joints `"-19 19"`.

**The joint RANGES have been imported into the lean model; the TORQUE limits have
not.** Measured on the built `Lean_H12_Magpie.xml`:

| actuator | lean model | CL_Assets (authoritative) | |
|---|---|---|---|
| shoulder pitch / roll | ±32 | **±40** | stale |
| shoulder yaw / elbow | ±14.4 | **±18** | stale |
| wrist roll / pitch / yaw | ±9.5 | **±19** | stale, **2x** |

Ranges that DID import correctly: knee, torso, elbow, wrist_roll all match CL exactly.

**Consequence for this project:** S2.4's claim — "the bracing arm's torque ceiling is
tiny and is very likely the binding constraint in the whole formulation" — was
computed against limits that are 25 % low at the shoulder and elbow and **100 % low at
the wrist**. The arm is materially stronger than I reported. That argument must be
re-run before it goes anywhere near the paper, and the palm/wrist results are the ones
most affected (the wrist error is the largest).

## S10.3 A concrete bug for Allen / the asset-unification work

Two things the limit import appears to have missed:

1. **`actuatorfrcrange` is not being propagated** to the lean models, only `range`.
   Everything downstream of `h1_2_base` still plans against the retired torque
   envelope.
2. **`right_shoulder_yaw_joint` range is L/R mirrored**: lean model `[-3.01, 2.66]`,
   CL_Assets `[-2.66, 3.01]`. The generator docstring flags exactly this swap for
   *left*_shoulder_yaw in the OLD base; it looks like the fix did not mirror to the
   right side.

## S10.4 Not done

Renders still unviewable (image Read blocked by hook timeouts for a third session).
Planning-rate check, gripper-orientation task, MJPC handoff sketch all still pending.
The 32-subset run used a coarse 8-target grid; the full 45-target grid over 32 subsets
is ~2 h and has not been run.

---

# SESSION 10b (2026-08-04) — CL torque override wired; re-runs launched

`contact_select.py` now honours `CL_ARM_TORQUE=1`, which overrides the lean model's
stale arm `actuatorfrcrange` with the CL_Assets / h12_safety_layer values
(shoulder 40, shoulder_yaw/elbow 18, wrist 19) inside `equilibrium_qp`. This is an
override rather than a model edit deliberately: the model belongs to
`asset-unification` and the real fix is to make the limit import propagate
`actuatorfrcrange` (S10.3), not to hand-patch a generated file.

Spot check at (1.00, 0.15, 0.95), subset `elbow+forearm`:

| | stale limits | CL limits |
|---|---|---|
| max abs(tau)/limit | 0.354 | **0.315** |
| effort | 0.361 | **0.291** |

~11 % lower peak utilisation, ~19 % lower effort. Direction is as expected; the
question is whether it changes which subset WINS.

Launched (background, will outlive the usage reset):
- `sweep5_cl_coarse.json` — 8 targets x 32 subsets under CL limits, direct A/B
  against `sweep5.json` from S10.1
- `sweep5_full_cl.json` — full 45-target grid x 32 subsets under CL limits, the
  definitive map over the enlarged candidate set (~2 h)

---

# SESSION 10c — user constraint: trunk contacts must be CONTROLLABLE and RECOVERABLE

> "we can use torso/hip contacts, but they're not necessarily desirable unless we can
> show that we can precisely control the contact and easily recover from it"

Static effort is silent on both. Built two tools to make the requirement measurable.

## S10c.1 `controllability.py` — force authority + release margin

- **Modulation band**: LP max/min of a contact's normal force over the feasible set
  (equilibrium + friction + torque limits). Wide band = force is a commandable DOF.
- **Release margin**: is the SAME pose still in equilibrium with the contact deleted?

Bug found and fixed while writing it: the friction pyramid was built by negating the
whole row, which pins |lam_t| TO the cone boundary instead of bounding it -- every LP
came back infeasible. Correct form needs two rows, `+lam_t - mu*lam_n/sqrt2 <= 0` and
`-lam_t - mu*lam_n/sqrt2 <= 0`.

At (1.00, 0.15, 0.95), subset `elbow+forearm+hip`, CL torque limits:

| contact | normal force | admissible band | release |
|---|---|---|---|
| elbow | 63.6 N | 0 - 478 N | releasable in place, peak 0.25 |
| forearm | 67.5 N | 0 - 432 N | releasable in place, peak 0.26 |
| **hip** | 66.2 N | **0 - 544 N** | releasable in place, peak 0.25 |

On force authority and static recoverability the hip scores BEST.

## S10c.2 `precision.py` — contact-PLACEMENT sensitivity, and the real answer

sigma_max of the contact point's position Jacobian over actuated joints; participation
ratio gives the effective number of joints carrying that sensitivity.

| site | sigma_max [m/rad] | placement error @1 deg | effective joints |
|---|---|---|---|
| hip | 0.05 | **0.9 mm** | **1.0** (torso only) |
| torso | 0.07 | 1.2 mm | **1.0** (torso only) |
| elbow | 0.46 | 8.0 mm | 2.6 |
| forearm | 0.49 | 8.6 mm | 2.5 |
| palm | 0.72 | 12.5 mm | 2.9 |

Naive reading: trunk contacts are ~10x MORE precisely placed. **That reading is wrong
and the number actually supports the user's instinct for a different reason.** Low
sensitivity means low AUTHORITY: the hip contact depends on effectively ONE joint
(torso, participation 1.0). It cannot be steered -- only the whole body can be placed
and the hip lands where it lands. Arm contacts are less precisely determined but carry
2.5-3 joints of redundancy with which to actively servo the contact.

**Sharp statement: the trunk contact is not a controllable contact, it is a consequence
of the body pose.**

## S10c.3 Proposed gate

Keep trunk sites in the candidate set but gate them on **steerability**: require >= 2
effective joints of contact-point authority. Hip (1.0) and torso (1.0) fail; elbow
(2.6), forearm (2.5) and palm (2.9) pass. This encodes the user's requirement as a
constraint rather than a preference, and it is computed, not asserted.

## S10c.4 What these metrics still cannot see

The release test is STATIC: it shows an equilibrium exists without the contact, not
that the transition into it is stable. The trunk carries most of the body mass, so the
gap between "an equilibrium exists" and "the robot can actually get there" is widest
exactly where the user is worried. Closing it needs a dynamic check (capture point /
reachability under the release manoeuvre), which this formulation cannot provide.

## Log

- 2026-08-02 21:34 MDT — brief created; no analysis performed yet by request.
- 2026-08-03 11:54 MDT — Session 1. Machine slept through the 00:35/05:35 wakeups;
  ran on the first live wake. icra2026 worktree + build up; paper read in full;
  lean task inventoried; mapping written (§S1.1–S1.6).
