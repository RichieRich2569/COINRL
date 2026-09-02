---
name: amortised-coin
description: Working history of the AmortisedCOINPPOAgent — what is implemented, what was removed and why, every experiment run and its numbers, the named failure modes, and the diagnostic tooling. Load before changing rl.py's amortised agent, before proposing a new mechanism, or when interpreting a pair-stream / pilot result.
---

# Amortised COIN-PPO: implementation and experiment log

Read this before touching `AmortisedCOINPPOAgent`. Its purpose is to stop
re-implementing mechanisms that were already built and deleted, and to stop
re-deriving conclusions that are already measured. Every table here is a real run;
where a number is unverified or superseded it says so.

**Last updated 2026-09-02.** 54 tests.

**Where this work lives.** Branch `amortised-coin-protection` on `origin`, forked from
`realtime-coinrl` at `99b1d73`. Developed in worktree `.claude/worktrees/ewc-encoder`.
The off-policy `V_z` line (`V_ξ(s,z)` replacing the PPO critic, plus `zvalue_probe.py`,
`plot_pair_run.py`, `coin_synth_test.py`, `model_equations.md`) lives **only** in the
working tree of `.claude/worktrees/offpolicy-zvalue` — uncommitted as of this writing.
Commit it before removing that worktree or it is gone.

**Keeping this file current.** It is the working log, so it is part of the deliverable,
not a byproduct: any commit that adds, removes or measures a mechanism updates the
matching section here in the same commit. A result that is not written down here has,
in practice, not been obtained — this session re-derived two mechanisms from scratch
and lost a day to a stale baseline `.npz` for exactly that reason.

---

## 0. Rules of engagement (from the author, not negotiable)

1. **Training-path changes need sign-off.** Propose first; build after Richard's word.
   `protect_heads` was built without it and reverted wholesale (`4762bf7` → `a71f039`).
2. **New mechanisms need either (a) a source paper doing the same thing, or
   (b) explicit permission.** Cite the paper in the docstring.
3. **The simplest architecture that uses COIN wins.** Every surviving component must
   have an ablation showing what breaks without it.
4. **Report every axis, never one.** Each intervention so far has traded one axis for
   another. A probe reports: task-A eval, task-A *oracle best-head*, task-B eval, K,
   code gap / d′, and per-term gradient norms. Eval and oracle differ — that
   distinction is what separates *routing* failure from *head destruction*, and the
   original gates could not tell them apart.
5. **Never trust a cached baseline `.npz` without checking its commit.**
   `models/pairstream_e2_mirror_t34_s1.npz` was used as a control for most of a
   session before it was noticed it predates `ba254f6`, which changed `prior_sd` and
   `kl_coef`. Several comparisons had to be retracted.

---

## 1. The architecture as it stands

### 1.1 The core idea (unchanged since `b3dfced`)

A `ContingencyEncoder` compresses a segment of transitions into a Gaussian posterior
over a **scalar** latent contingency `z`; a decoder gives `z` something to mean; COIN
consumes `z` as an observation and returns context probabilities; the agent acts under
a set of per-context PPO heads mixed by those probabilities.

**Encoder.** Per-transition factors, combined in natural parameters, so a prefix
posterior is a cumulative sum (all prefix lengths for free in one pass):

```
q(z | h_1:t)  =  N(0, prior_sd^2) · prod_{i<=t} N(z; mu_i, sigma_i^2)

precision:  lambda_t = 1/prior_sd^2 + sum_i 1/sigma_i^2
mean:       m_t      = (sum_i mu_i / sigma_i^2) / lambda_t
```

**Objective** (encoder + decoder, their own optimiser — the PPO step must never touch
them):

```
L_enc = L_dyn + kl_coef · KL( q(z|h) || N(0, prior_sd^2) )
L_dyn = MSE( f_psi(s, a, z), s' )        [+ reward output if encoder_reward]
```

**`z` is stop-gradiented out of PPO entirely.** This is the founding decision and it
is well-motivated: the pre-amortised `COINPPOAgent` trained the encoder through the
PPO loss via a differentiable responsibility, whose derivative is *identically zero
whenever the heads agree* — i.e. at onset and at every context birth, since new heads
are deep copies. Any proposal to "just backprop the value through COIN" is that dead
signal returning; see §3 `value_coef` for the form in which it was legitimately
revived.

**Segments and trials.** One rollout is S contiguous fixed-length segments, each its
own COIN trial. The *episode* is carried across segment boundaries (`_carry`), so a
task switch is a perturbation of an ongoing episode, not a reset.

### 1.2 Current defaults

| parameter | default | note |
|---|---|---|
| `prior_sd` | 0.5 | states COIN's operating envelope: codes live within ~±1 |
| `kl_coef` | 1e-4 | the *only* containment since `ba254f6`; calibrated by `beta_window_check.py` |
| `encoder_lr` | 3e-4 | |
| `replay_capacity` | 128 segments | reservoir (Vitter R), not a window |
| `decoder_lr_ratio` | 1.0 | >1 slows the decoder → gauge control at the source |
| `value_coef` | 0.0 | off by default |
| `encoder_reward` | False | |
| `observe_value` | False | 2-D COIN observation |

⚠️ **`figures.ipynb` cell 37 had `KL_COEF = 0.0`**, a leftover from the tanh-bounded
architecture. With the PEARL bottleneck that is *no containment at all*. Fixed in the
pilot driver (`run_amortised_pilot.py --kl-coef 1e-4`); check it before any notebook run.

### 1.3 Code map (`rl.py`, current worktree line numbers — verify, they move)

| concern | symbol |
|---|---|
| agent | `AmortisedCOINPPOAgent` (~2817) |
| acting weights, `act_gate` | `_policy_weights` (~3023) |
| learning gate | `_learn_gate_mask`, `_gate` (~2182–2210, used in `_mixed_logits`/`_mixed_value`) |
| encoder objective | `_update_encoder` (~3204), `_dyn_loss`, `_kl_to_prior` |
| value-gradient term | `_encoder_value_loss` (~3481) |
| COIN-centre anchor | `_encoder_anchor_loss` (~3252), `_note_context` (~3283) |
| encoder EWC | `_ewc_params`, `_ewc_penalty`, `consolidate_encoder` (~3378–3438) |
| head EWC | `_head_params`, `_ewc_head_penalty`, `consolidate_heads` (~3302–3377) |
| main loop | `train_step` (~3544) |
| eval | `evaluate_identifying` (~3892) |

`realtimecoin` is imported lazily so `rl.py` stays importable without it. Convert
query vectors with `rl.coin_context_vector` / `coin_context_trace` — never hand a raw
global-frame vector to an agent.

---

## 2. Timeline: what was added, when, and whether it survived

| commit | date | added | status |
|---|---|---|---|
| `b3dfced` | 08-26 | the amortised agent: encoder + decoder, `L_dyn + β·KL`, `z` stop-gradiented, COIN observes segment-final `z` | **live** |
| `3ceacbe` | | per-step context weighting (weights vary within a segment, not segment-constant); segment-size study | **live** |
| `8159a1f` | | Fig3 v2 campaign: MirrorCartPole swap, segment replay reservoir, episode carry-over, `decoder_lr_ratio`, `value_coef`, `encoder_reward`, `observe_value` (2-D COIN), + seven encoder/COIN mechanisms | mostly live; see §3 |
| `c54e8c9` | 09-01 | explicit `RuntimeError` on empty replay sample | live |
| `666aee2` | 09-01 | value observation rides the **raw** reward channel (shaped MC returns inflated it ~+0.5) | live |
| `ea60b74`, `9b0dad9` | 09-01 | episodic value-gradient steps (one encoder step per completed episode) | **removed** `8a2759a` |
| `4762bf7` | 09-01 | `protect_heads` — value-surprise segments sit out the PPO update | **reverted** `a71f039` (no sign-off) |
| `8a2759a` | 09-01 | the strip — see §4 | — |
| `ba254f6` | 09-01 | PEARL containment: tanh `z_scale` deleted, KL is sole containment, `prior_sd` 0.5 / `kl_coef` 1e-4 | **live** |
| *(uncommitted, this worktree)* | 09-02 | encoder EWC, head EWC, OWL gates, ported COIN-centre anchor | **experimental**, see §5 |

---

## 3. Live mechanisms, and the argument for each

- **Segment replay reservoir** — the encoder sees the whole stream, not the current
  block. Measured: prevents forgetting (dyn-loss ratio 18.5 → 3.7) but **does not stop
  code drift** (drift 0.80, the worst of any arm). The old docstring claim that the
  reservoir "stops the latent drifting with the curriculum" is wrong as stated.
- **Episode carry-over** — a task switch mid-episode is the biologically faithful
  perturbation and avoids handing the agent a free reset cue.
- **`decoder_lr_ratio`** — gauge control at the source: a slower decoder cannot chase
  a relabelled latent as fast as the encoder can relabel it. Preferable in principle to
  a penalty on the codes.
- **`value_coef`** — the old value gradient revived as an *auxiliary encoder term*:
  responsibilities recomputed differentiably from the encoder posterior and COIN's
  per-context Gaussians (COIN quantities constant), head values mixed under them (head
  outputs detached), mismatch against PPO return targets backpropagated into the
  **encoder alone**. Its gradient is zero while heads agree — which is exactly where
  `L_dyn` is strong — and nonzero where different heads value the same states
  differently, which is exactly where `L_dyn` is blind. Complements, not alternatives.
  *Ablation says it is removable* (§6.3).
- **`encoder_reward`** — raw reward enters the encoder *features* and is a decoder
  **target**, never a decoder input (an input revives the identify-from-anything-but-`z`
  shortcut). For the paper: this grants reward-as-context, the same concession the
  recurrent-PPO baseline already has.
- **`observe_value`** — COIN observes `(z, mean episodic return)` via realtimecoin's MD
  pipeline. Rationale: performance becomes part of what a context *is*, so a task
  parking at an established code still collapses the return dimension many sigma below
  that context's history → birth *before* the old head is overwritten
  (detect-before-adapt inside the inference model). Return read only from episodes that
  END in the segment; otherwise `(z, nan)` and the MD pipeline conditions on `z` alone.
  Requires the notebook to build `RealTimeCOIN(state_dim=2, process_noise_covariance=
  diag([sigma_process^2, value_process_noise^2]), ...)`.
- **PEARL containment (`kl_coef`)** — the mean is unbounded; β must sit between
  posterior collapse and unconstrained wander. `beta_window_check.py`: β=1e-4 gives
  codes spanning ±1 at 12× separation; β=1e-3 collapses; β=0 wanders.

---

## 4. Removed / rejected — do not re-add without new evidence

All removed in `8a2759a` ("strip to the minimal stack", −1097 lines) unless noted:

stamped anchor · rail hinge · channel noise · residual decoder · episodic value steps ·
quantile · repulsion + detector · dispersion penalty · EMA / centroid / distance
anchors · balanced replay · FiLM conditioning · `dyn_obs_norm` · `z_scale` tanh bound
(`ba254f6`) · `protect_heads` (`a71f039`, rejected as a hard intervention in the PPO
data path).

Two of these keep resurfacing in new clothes:

- **Anchors.** Every anchor variant pulls codes toward a reference. They all trade
  separation for stability. The one anchor with a positive result (COIN-centre, coef
  0.03) is **configuration-dependent** — see §6.4.
- **Bounding `z`.** The tanh bound was replaced deliberately by the KL bottleneck.
  If `z` leaves the ±1 envelope the fix is β, not a clamp.

---

## 5. Experimental additions (uncommitted, worktree `ewc-encoder`)

- **`ewc_coef` / `ewc_protect_decoder`** — diagonal empirical Fisher of the same
  `L_dyn + β·KL` objective, snapshots accumulate; `consolidate_encoder()`.
- **`ewc_head_coef`** — Kirkpatrick's own target: Fisher from `∂ log π(a|s)/∂θ` on
  replayed `(s,a)` (recoverable from encoder feats — `feats[:, :obs_dim]` and
  `feats[:, od:od+ad].argmax(-1)` — so no extra storage). Consolidated **on COIN
  births**, applied in the PPO minibatch loop (heads have their own optimisers).
- **`learn_gate` (off/thresh/argmax)** — forward-preserving gradient gate
  `_gate(x, keep) = keep·x + (1−keep)·x.detach()`.
- **`act_gate` (off/argmax)** — OWL hard routing at *acting* time.
- **`anchor_coef` / `anchor_rho_min`** — pull replayed segments of **inactive**
  contexts back to a snapshot of that context's COIN centre, taken only while the
  context was confidently responsible (ρ ≥ 0.8). Deliberately not the live centre and
  deliberately not the active context: anchoring the active segment closes a loop with
  COIN whose fixed point is *any* current agreement, including a wrong one.

Separate worktree `offpolicy-zvalue` (not merged): off-policy TD(0) grounding
`L = E_D[(V_ξ(s,z) − y)²]`, `y = r/κ + γ(1−d)V_ξ̄(s', sg[z])`, Polyak target, κ=200,
with `V_ξ` **replacing** the PPO critic.

---

## 6. What has been measured

### 6.1 Against the Figure-3 baselines (final-checkpoint eval, mean over seeds)

| method | mean | MountainCar | Acrobot | FlatMC | CartPole | MirrorCP |
|---|---|---|---|---|---|---|
| OWL | **−13.3** | −154 | −130 | −78 | **153** | 143 |
| LLIRL | −43.4 | −161 | −143 | −76 | 33 | 130 |
| recurrent PPO | −61.6 | −200 | −117 | −53 | 16 | 46 |
| single PPO | −63.0 | −200 | −170 | −169 | 62 | 162 |
| EWC | −65.6 | −112 | −191 | −64 | 11 | 29 |
| **ours** | −73.8 | −200 | −196 | −200 | 28 | **200** |

Task 0 across checkpoints — the decisive column:

| method | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| EWC | −111 | −111 | −110 | −111 | **−112** |
| OWL | −154 | −154 | −154 | −154 | −154 |
| LLIRL | −111 | −114 | −161 | −161 | −161 |
| single PPO | −111 | −148 | −178 | −186 | −200 |
| **ours** | −156 | −154 | −200 | −200 | −200 |

**Diagnosis: we are not short of learning capability, we are short of protection.**
Best single-task result in the table (200 on MirrorCartPole) *and* the worst retention
curve. Plasticity is fine; stability is absent.

### 6.2 Full pilot, current code (`models/amort_pilot_s0.npz`, seed 0, 49 min)

| task | 300 | 400 | 450 | 500 | 550 rollouts |
|---|---|---|---|---|---|
| MountainCar | −110.2 | −105.0 | −165.2 | −200.0 | −200.0 |
| Acrobot | −200.0 | −128.1 | −138.3 | −200.0 | −173.9 |
| FlatMountainCar | −53.3 | −64.5 | −44.6 | −200.0 | −200.0 |
| CartPole | 24.7 | 9.4 | 9.5 | 180.5 | 9.5 |
| MirrorCartPole | 25.9 | 36.8 | 35.5 | 9.4 | 39.5 |
| **mean** | −62.6 | −50.3 | −60.6 | −82.0 | **−105.0** |
| oracle-routed | −110.2 | −116.5 | −116.0 | −89.0 | −102.5 |

K final 4; `z` range −1.53 … +3.93 (outside the ±1 envelope). Every task is learned
and then lost. Oracle ≈ actual, so at this configuration the heads are genuinely
destroyed, not merely mis-routed.

### 6.3 Pair-stream ablations (mirror pair, 25+25, seed 1, `--no-inject --probe-switch`)

`oracA/oracB` = best head under oracle routing.

| arm | evalA | evalB | oracA | oracB | K |
|---|---|---|---|---|---|
| `act_gate=argmax` + head-EWC 1e6 | 138.1 | 9.2 | 200.0 | 64.8 | 2 |
| − `act_gate` | 200.0 | 27.9 | 200.0 | 29.1 | 2 |
| − `value_coef` | 200.0 | 16.6 | 200.0 | 21.1 | 3 |
| − `encoder_reward` | 200.0 | 9.5 | 200.0 | 9.3 | 2 |
| head-EWC 1e5 | 9.5 | 9.2 | 200.0 | 61.5 | 3 |
| head-EWC 3e5 | 200.0 | 22.7 | 200.0 | 24.6 | 2 |

Reading: **`value_coef` and `encoder_reward` are removable** (retention unaffected).
**`act_gate` is not** — it roughly doubles task B (65 vs 21–29). Coefficients
1e5/3e5/1e6 all retain A, none let B learn; 1e4 gives neither.

### 6.4 Full-length arms — the hypothesis that failed (2026-09-02)

Hypothesis: B's shortfall under head EWC is a *slower start*, because in unprotected
runs mirror reached 200 only by hijacking CartPole's already-trained head, and
protection removes that shortcut. **Refuted.** At 50+50:

| arm | evalA | evalB | oracA | oracB | K | retention gate |
|---|---|---|---|---|---|---|
| `ab_long_s1` | 11.4 | 9.2 | 200.0 | 17.1 | 2 | pass |
| `ab_long_s2` | 9.2 | 21.3 | 14.1 | 20.0 | 3 | **fail** |

B is *worse* at 50+50 (17) than at 25+25 (65). Seed 2 never learned block A at all
(20 at the end of the block), so the recipe is **seed-fragile**. Seed 1 exposes a
distinct failure: the CartPole head **survives** (oracle 200) but evaluation routes
CartPole to the mirror head at w=0.91 — retention lost at the *routing* stage, not the
parameter stage. Head EWC as configured is not the answer; the next target is COIN's
eval-time routing.

### 6.5 Encoder EWC vs replay (`ewc_probe.py`, PPO-free, ~35 s, seed 0)

| arm | forget | drift | gap |
|---|---|---|---|
| replay (default) | 3.70 | 0.804 | 0.295 |
| none | 18.47 | 0.186 | 0.114 |
| ewc 1e8 | 1.68 | 0.142 | 0.094 |
| **replay + ewc 1e8** | **1.18** | **0.090** | 0.111 |
| encoder-only ewc 1e8 | 18.77 | 0.006 | 0.055 |

1. **Encoder EWC alone is useless** — pinning the encoder (drift 0.006) still forgets
   as badly as no protection, because the **decoder** re-fits and takes task A's
   dynamics with it. Protect both.
2. Replay and EWC are **complements**: replay supplies old *data*, EWC old *parameters*.
3. The coefficient must be ~1e8 — this objective's Fisher is ~1e-10. Far outside the
   usual EWC range; state it explicitly in any write-up.
4. The COIN column of this probe is **uninformative** (K=1 everywhere): random actions
   + 400 encoder steps do not build codes COIN can split.

### 6.6 Older arm table (mirror pair, seed 1, oracle probe)

| arm | div | sep | K | CartPole oracle | Mirror oracle |
|---|---|---|---|---|---|
| baseline (current code) | pass | pass | 2 | {9, 9} | {200, 181} |
| `learn_gate=argmax` | pass | pass | 2 | {9, 9} | {200, **9**} |
| `act_gate=argmax` | pass | pass | 2 | {9, 9} | {200, **200**} |
| act + learn gate | pass | pass | 2 | {9, 9} | {200, 200} |
| encoder EWC 1e6 / 1e7 | fail | fail | 3 | — | 200 |
| encoder EWC 1e8 | fail | pass | 2 | — | **9.5** (latent frozen) |
| `stationary_sd 1` alone | fail | fail | 6 | {9,10,9,9,9,9} | {194,9,9,200,36,200} |
| act_gate + `stationary_sd 1` | pass | pass | **10** | all 9 | all 200 |

**Across ~15 arms in both worktrees, exactly one configuration ever preserved task A
by anchoring:** the COIN-centre anchor at 0.03 in the *off-policy* stack (CartPole eval
122, oracle 135). Ported to this worktree's base at the same coefficient it gives
CartPole 9, alone and with `act_gate`. Its success therefore depended on something else
in that stack — candidates in order: `--no-inject`, the off-policy TD grounding + V_z
critic, `recent_frac 0.5`, `stationary_sd 1.0`.

---

## 7. The named failure modes

Keep these separate. Conflating them produced most of the wasted effort.

1. **Mixture identifiability.** `_mixed_logits` returns `Σ_c w_c · logits_c`, so PPO
   constrains only the **weighted sum**. Heads sharing responsibility become jointly
   meaningful and individually meaningless — a block whose mixture trains to 200 leaves
   every one of its heads scoring 9 alone, and evaluation (near single-head routing)
   reads the individual head. **Fixed by `act_gate=argmax`** (OWL hard acting routing):
   mirror then scores 200 *alone*, which no mixture-trained head ever does. Note that
   gating the gradient alone (`learn_gate`) is only half of OWL — the forward pass is
   still a mixture, so the trained head only learns to correct the mixture.
2. **Detection lag.** COIN does eventually birth a context for an arriving task, but in
   the rollouts before the birth the new task runs on the established task's head and
   overwrites it. Proven by `--probe-switch`: at the END of block A the heads score
   `{0: 200, 1: 9}`, so block A really does produce a competent head and all the loss
   is in block B's arrival window. **This is what head EWC protects.**
3. **Routing capture / decoy contexts.** An idle context's coordinate decays toward its
   stationary prior; the arriving task captures it instead of birthing (ρ flips
   `[0,1] → [1,0]` at the switch with no birth). Near-tied ρ makes argmax flicker so
   *both* heads get overwritten. Hard routing makes this **worse** (no mixture to
   dilute); a learned drift prior converts capture into proliferation (K=10, all ten
   heads become mirror). **Unsolved.**
4. **Eval-time mis-routing** (§6.4, seed 1). The head is intact but `z`-routing sends
   the task to the wrong one. Independent of 1–3 and currently the largest single loss
   in the pilot's better checkpoints. **Unsolved** — see B2 in §9.
5. **Gauge / coordinate drift.** One scalar coordinate, one shared decoder ⇒ identity
   lives in a coordinate, and a coordinate has a gauge. Codes translate together, COIN's
   stored centres go stale, a new task settles on a vacated coordinate. The anchor and
   the EWC penalty are both *repairs* to this; LLIRL's per-context decoders would
   *remove* it (§9, B4).
6. **Seed fragility.** `ab_long_s2` never learned block A. Any conclusion from a single
   seed is provisional — this bit twice.

---

## 8. Tooling

Run everything with the repo venv; plain `python` is not on PATH in the Bash tool:

```
C:\Users\richa\Documents\PhD Projects\COINRL\.venv\Scripts\python.exe
```

| script | what it does | cost |
|---|---|---|
| `pair_stream_test.py` | the workhorse: two-task blocked stream, three gates, oracle probe | minutes |
| `ewc_probe.py` | PPO-free encoder forgetting/drift/separation sweep | ~35 s |
| `beta_window_check.py` | calibrates `kl_coef` against the ±1 envelope | fast |
| `percontext_decoder_probe.py` | per-context vs shared decoder separation | fast |
| `smoke_amortised.py` | minimal construction + one train_step | seconds |
| `run_amortised_pilot.py` | full pilot, **streaming**; execs `figures.ipynb` cell 37's body so it cannot drift from `TRAIN_AMORTISED` | ~50 min |
| `show_pilot.py`, `summarise_arms.py` | tabulate a pilot / a set of arms | instant |

`pair_stream_test.py` flags: `--ewc-coef --ewc-heads-coef --ewc-max-tasks --learn-gate
--learn-gate-thresh --act-gate --anchor-coef --stationary-sd --no-inject --probe-switch
--value-coef --no-encoder-reward --no-observe-value`.

**Gates.** (1) divergence — code gap grows; (2) eval-separation — distinct dominant
heads; (3) retention — *some* head still scores >140 on task A under **oracle**
routing. Gate 3 deliberately separates head survival from routing: a pass with a low
self-identified eval means routing lost it, which is a different bug.

**Running long jobs.** Harness-tracked long silent runs have been killed; detached
`Start-Process` with `> log 2> err` survives. Always tee a log — the notebook route
emits nothing for ~2 h, which is indistinguishable from a hang and is why the pilot was
killed twice.

---

## 9. Open candidates (from `borrow_from_baselines.md`)

- **B2 — bandit head selection at evaluation** (OWL's UCB1 over episode returns).
  A *performance-based* selector, immune to a stale code; fails in a different
  direction from `z`-routing. Directly targets failure mode 4, which §6.4 has now
  promoted to the top of the list. Needs checkpoint saving (`pair_stream_test` saves
  none); then it re-scores existing runs in seconds. **Not implemented.**
- **B4 — per-context decoders** (LLIRL). Removes failure mode 5 at the root: identity
  becomes a likelihood comparison between models rather than a distance in a shared
  latent. Cost: it weakens the paper's story (the amortised scalar `z` feeding COIN
  *is* the contribution) and scales linearly in contexts. There is a cheap offline
  probe (~80 lines, ~1 min, no RL) that says whether it is worth it. **Not implemented.**
- **Retention / drift priors.** The author's note: earlier (pre-real-time) models gave
  each context its own stationary attractor via a `1/(1−a)²` factor in the drift prior.
  Directly relevant to failure mode 3. The original branch has the reference
  implementation.
- **Head-EWC schedule.** Do not keep sweeping the coefficient — 1e4/1e5/3e5/1e6 are all
  measured and none solve B. If revisited, change the *schedule* (what triggers
  consolidation) rather than λ.
