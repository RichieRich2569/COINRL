# What the Figure-3 baselines do better, and how to test borrowing it

## 1. The evidence

Final-checkpoint eval return, mean over seeds (`models/fig3_*_s*.npz`):

| method | mean | MountainCar | Acrobot | FlatMC | CartPole | MirrorCP |
|---|---|---|---|---|---|---|
| OWL | **−13.3** | −154 | −130 | −78 | **153** | 143 |
| LLIRL | −43.4 | −161 | −143 | −76 | 33 | 130 |
| recurrent PPO | −61.6 | −200 | −117 | −53 | 16 | 46 |
| single PPO | −63.0 | −200 | −170 | −169 | 62 | 162 |
| EWC | −65.6 | −112 | −191 | −64 | 11 | 29 |
| **ours (amortised)** | −73.8 | −200 | −196 | −200 | 28 | **200** |

Task 0 (MountainCar) across the five checkpoints — i.e. what happens to the first
task as later blocks train. **This is the decisive column:**

| method | ckpt 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **EWC** | −111 | −111 | −110 | −111 | **−112** |
| OWL | −154 | −154 | −154 | −154 | −154 |
| LLIRL | −111 | −114 | −161 | −161 | −161 |
| single PPO | −111 | −148 | −178 | −186 | −200 |
| **ours** | −156 | −154 | −200 | −200 | −200 |

### The diagnosis

Our architecture is **not short of learning capability — it is short of protection.**
We post the best single-task result of any method in the table (200 on MirrorCartPole,
against OWL's 143 and single-PPO's 162) and simultaneously the worst retention curve,
collapsing task 0 from −156 to −200. Plasticity is fine; stability is absent. That is
consistent with everything the pair-stream work found: separation and birth now work,
and task A's head still ends up destroyed.

So the mechanisms worth borrowing are the *protection* mechanisms, and the two methods
with flat retention curves — EWC and OWL — achieve it in completely different ways
that are **independent and stackable**:

- **EWC** protects *parameters* (a Fisher-weighted quadratic pull toward anchored weights).
- **OWL** protects *by construction*: only the routed head is ever updated, so the other
  heads cannot be touched at all.

Note EWC pays for its stability with plasticity (11 and 29 on the last two tasks, the
worst in the table), which is the same trade-off the encoder-side EWC probe measured
(`ewc_probe.py`: better retention and lower drift, ~1/3 less code separation). Any
borrow has to be measured on both axes, never one.

---

## 2. Four borrowable mechanisms

### B1 — OWL's hard learning routing *(highest value, smallest change)*

*What OWL does.* One head per task; each rollout is routed to exactly one head and
**only that head is updated**. Forgetting is zero by construction.

*What we do.* `_mixed_logits` sums `w_c · policy_c(obs)`, so **every head with non-zero
weight receives gradient in proportion to its weight**. The same weight vector serves
as a behavioural mixture *and* as a credit assignment. Measured directly in the
off-policy worktree: during the diffuse phase (max acting weight 0.27) heads 1 and 8
both learned MirrorCartPole to 150–166 *without ever being the responsible context*.

*The borrow.* Decouple acting from learning. Keep the soft mixture for behaviour;
gate the PPO update by responsibility — a hard argmax (full OWL), or a threshold, or a
sharpened `w^k` renormalised. COIN itself gates learning by responsibility, so this
also makes the agent more faithful to the model it is built on, not less.

*Probe.* Mirror pair, 25+25, seed 1, three arms: soft (current) / threshold 0.1 /
argmax. **Metrics:** task-A eval, oracle best-head on task A, task-B eval, K.
**Pass:** task-A oracle stays above 70% of its in-block training return with task-B
eval unharmed. **Status: not implemented** (~20 lines in `_logp_entropy` and the PPO
minibatch loop).

### B2 — OWL's bandit head selection at evaluation

*What OWL does.* With the label withheld, a fresh UCB1 bandit over episode returns
picks the head, per evaluation cell.

*Why it matters here.* Our evaluation routes purely on `z`. When the code drifts,
`z`-routing sends CartPole to a head that cannot do CartPole — measured: eval picked
head 7 (w=0.83) while the oracle best head was a different one entirely. A bandit is a
**performance-based** selector, completely independent of the encoder, so it cannot be
fooled by a stale code. It is a fallback that fails in a different direction from ours.

*The borrow.* At evaluation only (training untouched): run UCB1 over instantiated heads
for the first N episodes, then commit. Optionally seed the bandit's prior with the
`z`-marginal so it starts from COIN's belief and only overrides it under evidence.

*Probe.* Re-evaluate **existing checkpoints** — no retraining. Compare `z`-marginal
routing / UCB1 / `z`-seeded UCB1 on the same trained agent. **Pass:** bandit routing
recovers the oracle-head return where `z`-routing does not. **Status: not implemented**
(needs an agent checkpoint; `pair_stream_test` currently saves none — add a
`torch.save` and this probe costs seconds per arm).

*Caveat for the paper:* the bandit needs many episodes to identify, and it makes
evaluation adaptive rather than single-shot. Report it as an ablation, not as the
headline routing.

### B3 — EWC on the context heads

*What EWC does.* Anchors weights with a Fisher-weighted quadratic; flattest retention
curve in the table (−111 → −112 over the whole stream). Note the baseline computes its
Fisher on `∂ log π/∂θ` over the **policy**, and anchors **at the end of each block**.

*What we tested.* EWC on the **encoder/decoder** (`ewc_probe.py`) — a different target.
It works (forget 18.5 → 1.18 combined with replay) but does nothing for the heads,
which is where the pair-stream damage actually is.

*The borrow.* Apply the baseline's own recipe to our per-context heads, anchored when
COIN births a context rather than at a known block boundary.

*Probe.* Mirror pair, arms: none / EWC-heads / EWC-heads + B1 gating. **Pass:** task-A
oracle retained; and specifically whether B1 makes B3 redundant — if hard routing
already gives zero interference, the Fisher penalty is unnecessary complexity.
**Status: not implemented.**

### B4 — LLIRL's per-cluster dynamics models *(the redesign option)*

*What LLIRL does.* Each CRP cluster owns its **own** dynamics model `f_k(s,a) → s'`, and
cluster identity is inferred from which model predicts the data best.

*Why this is the deep one.* We have **one** decoder conditioned on a scalar `z`, so task
identity lives in a *coordinate* — and a coordinate has a gauge. Every failure the
pair-stream work traced comes from that gauge moving: codes translating +2 together, COIN's
stored centres going stale, an arriving task settling on a vacated coordinate. With
per-context decoders there is no shared coordinate to translate, and identification
becomes a likelihood comparison between models rather than a distance in a latent space.
The COIN-centre anchor and the EWC penalty are both *repairs* to the gauge problem;
this removes it.

*The cost.* It weakens the paper's story — the amortised scalar `z` feeding COIN is the
contribution, and per-context decoders make `z` much less central. It also scales
linearly in contexts. Treat as a fallback if the gauge repairs plateau.

*Probe.* Offline and cheap, no RL: collect segments from both tasks; fit (a) one
z-conditioned decoder, (b) one decoder per task. Measure held-out prediction and, more
importantly, whether per-model likelihood separates the tasks more sharply than the
z-distance does. **Pass:** likelihood margin between models exceeds the z-code d′.
**Status: not implemented** (~80 lines, ~1 min to run).

### Not worth borrowing

Recurrent PPO is the worst retainer *and* the worst learner (task 0 flat at −200 —
it never learned it). Nothing to take.

---

## 3. Suggested order

1. **B1** — biggest measured effect (the leakage is documented), smallest change,
   and it may subsume B3.
2. **B4 offline probe** — an hour of work, and it tells us whether the whole
   gauge-repair line is worth continuing before we invest further in it.
3. **B2** — needs checkpoint saving; useful as an ablation and a diagnostic even if
   it does not become the headline mechanism.
4. **B3** — only if B1 proves insufficient.

## 4. What every probe must report

The recurring lesson of this work is that single-axis results mislead — every
intervention so far has traded one axis for another, and the gates as originally
written passed while the agent was quietly broken. So each probe reports all of:

- **task-A eval** and **task-A oracle best-head** (routing failure vs head destruction —
  these are different failures and the gates cannot tell them apart);
- **task-B eval** (plasticity — did protection cost learning?);
- **K** at the end (spurious births);
- **code gap / d′** (separability);
- **per-term gradient norms** (the balance actually achieved, not the one intended).
