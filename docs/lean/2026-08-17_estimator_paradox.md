# The v3/v4 estimator reversal is the simulator's feet, not the estimator

**2026-08-17 · S21 · measured on the lean model, in-process MuJoCo, closed loop**

## The reported paradox

The same v4 estimator (v3 + a world-referenced auxiliary: AprilTags / FAST-LIO
on `rt/aux_odom`) is reported as:

| | v3 | v4 |
|---|---|---|
| simulation | good | **poor** |
| real robot | — | **good** |

Which reads as incomprehensible, because with no aux publisher running **v4 is
byte-identical to v3** — its own selftest asserts this. So the entire
difference is what fusing a world-referenced measurement does, and the
question is why that helps on hardware and hurts in sim.

## The mechanism was already written down

`base_estimator_node_v4.py` carries a note dated 2026-07-30 that describes the
failure exactly, and quantifies it:

> with healthy planted feet, aux re-references the estimate to the WORLD, so
> real foot creep (invisible+harmless to feet-referenced leg odom, which the
> controller's trim/balance loops were tuned on) enters the state as apparent
> drift → the trim leans the robot to "cancel" it → forced lean / v1-style
> foot shuffle (**v3 198 s vs v4-aux ~50 s stands**).

Leg odometry *defines* the planted foot as world-fixed. A foot that slides is
therefore invisible to it — and harmless, because the balance loops were tuned
against that same reference. A world-referenced aux sees the slide and calls
it base drift. Fuse it and the controller starts chasing a phantom.

That explanation only bites if the foot actually slides. So: measure it.

## What the foot actually does

Method: freeze the contact centroid as a **material point** of the foot and
follow it. A foot rotating about a planted sole then reads zero, which is the
whole point.

**This matters — the naive measure is wrong.** `d.xpos[ankle_roll_link]` is
the ankle joint origin, well above the sole, so it translates whenever the
ankle rotates. It reported 17–20 mm of "creep" on a 4 s brace where the true
material-point figure is under 1 mm/s. The tell was that
`mjOption.noslip_iterations` changed the number by nothing: a solver knob that
does not move a quantity is evidence the quantity was not solver drift.

### Brace+reach, 4 s — the foot does *not* creep

| | left | right |
|---|---|---|
| material point @ 1 s | 2.87 mm | 6.81 mm |
| @ 3.96 s | 3.03 mm | 7.08 mm |
| **rate over the last 3 s** | **0.05 mm/s** | **0.09 mm/s** |

An initial elastic take-up as load arrives, then a plateau. **Bounded give,
not drift.** Over a 4 s maneuver there is nothing here for an aux to disagree
with.

### A 60 s held stand — the feet creep, and they splay

| | left | right |
|---|---|---|
| @ 15 s | 4.21 mm | 7.73 mm |
| @ 30 s | 5.14 mm | 13.32 mm |
| @ 45 s | 6.41 mm | 19.03 mm |
| **@ 60 s** | **26.94 mm** | **31.56 mm** |
| final dy | **+24.53 mm** | **−30.94 mm** |

Monotone, and the two feet go in **opposite** y directions: the stance widens
by ~55 mm in a minute. Sustained rate ≈ 0.5 mm/s per foot. Extrapolated to the
198 s stand in the note above, that is ~10 cm of stance splay — accumulated by
a robot that is, as far as leg odometry is concerned, standing perfectly still.

The regime is what decides it. The 4 s brace has no creep; the long stand has
plenty. And the long stand is exactly the regime the reversal was reported in.

## Why this is a simulator artifact and not physics

While the feet are loaded and moving, the tangential-to-normal force ratio has
median **0.19–0.20** against a friction coefficient of **μ = 1.0**, and
**zero** moving periods reach 95% of the cone. A Coulomb contact at a fifth of
its friction cone slips *exactly nothing*. The sliding is the regularized
contact model's compliance, not friction being overcome.

Two standard remedies were tested and neither helps, which is worth recording:

- `noslip_iterations` 0 → 10 → 30: 1.63 → 1.54 → 1.52 mm/s. Inside run-to-run
  variation. It is a velocity-level post-projection; this is position-level
  compliance.
- `impratio`: **the lean model already ships `impratio = 100`**, the stiff-
  friction setting. Dropping it to 10 makes slip *worse* (1.63 → 3.17 mm/s),
  which confirms it is the governing parameter and that 100 is already right.

So the creep is not a misconfiguration to be fixed. It is what this contact
model does, already at its best setting.

## What this resolves

The pattern follows without anything being wrong with v4:

1. In a long **sim** stand the feet creep ~0.5 mm/s each and splay.
2. Leg odometry reports none of it; the trim loop is consistent with that.
3. The world-referenced aux reports all of it, as base drift.
4. v4 fuses it → the trim leans to cancel a drift the simulator invented →
   shorter stands. **v3 wins in sim by being blind to a fiction.**
5. On the real robot feet do not splay 3 cm/minute standing still, so aux and
   leg odom agree and the aux only removes genuine drift. **v4 wins on
   hardware by being right.**

The existing `--aux-gate planted` default — fuse world-referenced aux only
while the feet are *not* trustworthy — is the correct mitigation, and this is
why. It is not a hack around a bad sensor; it is a guard against the one
regime where the simulator's reference frame and the robot's disagree.

## Caveats, stated plainly

- Measured on **this** repo's lean model and table scene, under the crocoddyl
  MPC. Allen's result is on the MJPC deploy stack with a different model and
  different contact parameters. The mechanism transfers; the numbers should be
  re-measured there before being quoted.
- The 60 s "stand" is croco_twin's `stand` plan held at its terminal node, not
  a dedicated standing controller.
- The real robot's foot creep was **not** measured — the asymmetry in step 5
  is inference from the friction-cone argument, not from hardware data. It is
  the one load-bearing claim here that is not measured, and it is worth an
  hour with a tape measure and a 200 s stand.

## The falsifiable prediction

Log `|aux_xy − legodom_xy|` over a long stand, in sim and on hardware. This
says sim diverges at roughly the splay rate (~0.5 mm/s, tens of mm/minute) and
hardware stays flat. If hardware diverges at the same rate, this explanation is
wrong and the cause is elsewhere.
