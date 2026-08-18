# The browser panel

`croco_twin --gui [PORT]` serves a panel on `http://127.0.0.1:8770/` and turns
the script into a **session**: it runs episodes until told to stop, so reset,
pause, the viewer and the task dropdown all have something to be relative to.
Without `--gui` it is still the one-shot script `twin_grid.sh` drives 26 times
in a row, and that batch behaviour is the default on purpose.

No dependencies. The WebSocket server is ~260 lines of socket code in
`croco/gui/ws.py`; the page is one static file.

## The three plots

**solve time vs the control period.** The dashed red line is the period. Marks
on the trace are the periods that crossed it. This is the plot that decides
whether the controller is deployable at all.

**cost, per term.** One trace per cost term in the *current* OCP, with a
checkbox each. The scale is the largest shown term, so hiding the big ones is
how you see the small ones move. Cost families are collapsed to one entry:
a keep-out set is per sampled point (`ko_<geom>_<point>`, 90 of them on a
keepout cell) and they are one physical intent, so they arrive as a single
`keepout` trace and a single slider. Left expanded they were 90 of 104
sliders, and the fifteen terms anyone reasons about were unfindable.

**wall period vs deadline.** What the loop actually achieved, against what it
owed. State age is the faint green second series. Free-running in process
there IS no deadline — which is why a free-run overrun count of zero means
nothing, and `--realtime 1.0` is what makes the number real.

Red vertical rules appear here when `--safety` is on: the periods the
h12_safety_layer's e-stop predicate held. **The first one is drawn solid and
the rest translucent**, because the layer's e-stop latches — everything to the
right of the first rule is a trajectory the layer would never have allowed.

## Controls

| | |
|---|---|
| **play / pause** | Freezes the loop and writes a damping hold. Paused periods leave **no video frames** — the recorder is an `on_step` hook and a paused period never calls it — so the render is the trajectory, not the wall-clock session. |
| **reset** | Puts the plant back at the plan's start **and rewinds the MPC** (`MPC.reset()`, 8.8 ms). Rewinding only one of the two was measured to drop the pelvis to 0.069 m: the window stays parked at the end of the plan while the robot is back at the beginning. In-process only — the twin owns its physics and has no reset channel. |
| **render** | Renders the last episode to mp4 and shows it in the panel. Off the control thread, after the run. |
| **viewer** | Opens/closes MuJoCo's passive viewer mid-session. Disabled when `MUJOCO_GL` is an offscreen backend (`egl`, `osmesa`) — the button says so rather than failing. |
| **task** | `brace+reach` / `stand` / `recover`. A task is a **solved plan** in the cell, not a set of weights: the phases differ by contact set and no weight can add a contact. Unsolved tasks are offered-but-disabled with the command that solves them. |
| **submode** | `single-shot` runs the plan once and idles. `hold` freezes the plan index at the last node and keeps solving — the brace stays braced. `automode` loops the available tasks back to back, chained. |
| **speed** | Sim seconds per wall second. Below 1× the solver gets wall clock the robot would not give it, so the overrun count stops being a deployment result — and the panel says so. |
| **reach target** | xyz entry boxes with nudge arrows. Applied **in place** to every node carrying the reach cost — no rebuild, and it survives a reset. The warm start still descends toward the original target, so small moves track and large ones want a re-solve. |
| **gripper roll** | Rotates the commanded orientation about a local axis, away from the one q\* already reaches. Needs the `reachRot` cost to exist — see below. |

## Cost weights

Log-scaled sliders, 1/100× to 100× the value the run started with. Edits are
applied **between periods** on the control thread (writing a weight into a
model the solver is reading is a data race whose symptom is a bad step, not a
crash), and every change is recorded in the run artifact — a retuned run that
does not say so is not reproducible. The header shows a warning while any
weight differs from the plan's.

Edits made while the session is idle are queued and applied at the next
episode's first period.

### Why `reachRot` may be missing

It is a cost, and a cost that does not exist in the built models cannot be
added to one — crocoddyl allocates per-cost data at construction. Every cell
in the certified S18 grid was solved with `w_reach_rot = 0`, so
`_reach_orientation` returned early and there is nothing to point anywhere.
Two fixes, both stated in the panel itself:

```bash
croco_twin ... --reach-rot auto      # rebuild with it, re-solves nothing
studies/solve_tasks.sh <cell>        # re-solve the cell with it
```

`auto` points the reference at the orientation q\* already reaches, so at the
default token weight of 1e-2 it changes the plan by ~4e-5 in cost — it exists
in order to be steerable, and the weight slider is what gives it authority.

## Late arrivals see the whole run

The panel keeps a backlog and replays it on connect, so opening the browser
after the maneuver has finished shows the maneuver rather than an empty page.
This was the first thing the panel got wrong: `broadcast` is push-only, and a
4 s run against a page opened at t≈0 left nothing to look at.
