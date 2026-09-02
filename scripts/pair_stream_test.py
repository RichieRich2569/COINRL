"""Parametrised miniature pair-stream test: task A block then task B block on the
minimal stack (2-D COIN observation + value term + reward features),
expert-bootstrapped heads, per-rollout probe-code traces, and a post-stream
evaluate_identifying (the real z-marginal eval) on both tasks. NOT a pilot -- a
targeted probe of birth / z-divergence / eval routing for one task pair. Set
FIG3_TASK4=gravflip in the environment to make task 4 the original InvertedCartPole.

Usage: python pair_stream_test.py --tasks 3 4 --rollouts 50 --out e1.npz
"""
import argparse
import sys
import time

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

import numpy as np

# MC-family tasks need the wide-rollout recipe; the rest train fine at fig3 size.
PRETRAIN_ITERS_DEFAULT = {0: 150, 1: 100, 2: 80, 3: 100, 4: 100}
PRETRAIN_KW_WIDE = dict(rollout_steps=8192, mini_epochs=10, mb_size=256)
PRETRAIN_KW_STD = dict(rollout_steps=2048, mini_epochs=10, mb_size=64)
CTX_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "novel"]
PROBE_SEGS, SEG_STEPS = 8, 256
EVAL_EPISODES = 30
KL_BETA = 1e-4     # PEARL bottleneck weight, from the beta-window check: codes span
                   # +-1 (the COIN envelope) with 12x task separation; collapse
                   # begins at 1e-3, unconstrained wander at 0


def pretrain_expert(task):
    import torch
    torch.set_num_threads(2)

    import rl
    from rl import PPOAgent
    from baselines import fig3_common as f3

    rl.seed_everything(100 + task)
    env = f3.make_task_env(task, seed=100 + task)
    agent = PPOAgent(env, **f3.PPO_KWARGS)
    kw = PRETRAIN_KW_WIDE if task in (0, 2) else PRETRAIN_KW_STD
    last = None
    for k in range(PRETRAIN_ITERS_DEFAULT[task]):
        r = agent.train_step(env, **kw)
        last = r["mean_episode_return"]
    env.close()
    print(f"[expert {f3.TASK_NAMES[task]}] final train return {last:.1f}", flush=True)
    return ({k: v.cpu() for k, v in agent.policy.state_dict().items()},
            {k: v.cpu() for k, v in agent.value_net.state_dict().items()},
            agent)


def collect_probe(agent_amort, expert, task):
    """Fixed on-policy probe segments: expert acts greedily on raw envs."""
    import torch

    import rl
    from baselines import fig3_common as f3

    pol = f3.GreedyPPOPolicy(expert)
    feats = []
    envs = f3.make_task_envs(task, PROBE_SEGS, seed=7000 + task,
                             max_episode_steps=200, train=False)
    with torch.no_grad():
        for env in envs:
            obs_t = agent_amort._flatten_obs(env.reset()[0])
            obs, act, nxt, rew = [], [], [], []
            for _ in range(SEG_STEPS):
                a = pol.act(obs_t.numpy())
                next_obs, r, done, trunc, _ = env.step(a)
                obs.append(obs_t.cpu())
                act.append(a)
                nxt.append(agent_amort._flatten_obs(next_obs).cpu())
                rew.append(float(r))
                if done or trunc:
                    next_obs, _ = env.reset()
                obs_t = agent_amort._flatten_obs(next_obs)
            feats.append(agent_amort._segment_features(obs, act, nxt, rew=rew))
    f3.close_envs(envs)
    return torch.cat(feats)


def eval_fixed_head(agent, env, head_index, n_episodes=10, max_steps=200):
    """Mean return with the acting weights PINNED to one head (oracle routing).

    Separates the two failures the gates cannot distinguish: if some head still scores
    on task A, the knowledge survived and only the ROUTING lost it; if no head scores,
    the head itself was overwritten.
    """
    import torch

    w = torch.zeros(agent.num_contexts, dtype=torch.float32, device=agent.device)
    w[head_index] = 1.0
    rets = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=12345 + ep)
            obs_t = agent._flatten_obs(obs)
            total = 0.0
            for _ in range(max_steps):
                a = agent._deterministic_action(obs_t, w)
                obs, r, done, trunc, info = env.step(a)
                total += (float(info.get("raw_reward", r))
                          if isinstance(info, dict) else float(r))
                if done or trunc:
                    break
                obs_t = agent._flatten_obs(obs)
            rets.append(total)
    return float(np.mean(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--rollouts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0,
                    help="stream seed: seed_everything, COIN rng, rollout env seeds"
                         " (experts and probe/eval seeds stay fixed across arms)")
    ap.add_argument("--ewc-coef", type=float, default=0.0,
                    help="EWC on encoder AND decoder, consolidated whenever COIN"
                         " births a context; 0 = the committed replay-only baseline."
                         " Needs ~1e8: this objective's Fisher is ~1e-10")
    ap.add_argument("--learn-gate", choices=("off", "thresh", "argmax"), default="off",
                    help="OWL-style credit-assignment gate: which heads may receive"
                         " gradient. Behaviour (the acting mixture) is unchanged")
    ap.add_argument("--learn-gate-thresh", type=float, default=0.1)
    ap.add_argument("--probe-switch", action="store_true",
                    help="probe every head on task A at the end of block A, before"
                         " task B arrives")
    ap.add_argument("--no-inject", action="store_true",
                    help="skip the expert-injection bootstrap of newly born heads."
                         " Injection is a diagnostic crutch: its one-rollout return"
                         " jump is itself a large surprise on COIN's value dimension"
                         " and has been measured to cause spurious births")
    ap.add_argument("--anchor-coef", type=float, default=0.0,
                    help="COIN-centre anchor: hold inactive contexts' replayed"
                         " segments at the centre COIN stored for them. Its gradient"
                         " must sit BELOW the dynamics gradient (~0.03 works; 0.3"
                         " collapses the codes below COIN's observation noise)")
    ap.add_argument("--stationary-sd", type=float, default=None,
                    help="loosen COIN's drift prior so each context LEARNS a drift"
                         " parking its stationary value d/(1-a) at its own coordinate:"
                         " prior_precision_drift = 1/((1-a)^2 sd^2). None keeps the"
                         " published prior, under which an idle context reverts to"
                         " ~0 in ~12 trials and becomes the decoy an arriving task"
                         " captures instead of triggering a birth")
    ap.add_argument("--act-gate", choices=("off", "argmax"), default="off",
                    help="OWL-style hard acting routing: one head acts alone, so each"
                         " head must be individually competent")
    ap.add_argument("--ewc-heads-coef", type=float, default=0.0,
                    help="Kirkpatrick's penalty on the context HEADS -- the Figure-3"
                         " EWC baseline's own target -- consolidated whenever COIN"
                         " births a context, to protect an established head through"
                         " the detection lag. Independent of --ewc-coef, because"
                         " encoder EWC was measured to hurt on a live stream")
    # Ablation switches for the components the committed agent turns on by default.
    ap.add_argument("--value-coef", type=float, default=1e-3,
                    help="responsibility-path value term; 0 removes it")
    ap.add_argument("--no-encoder-reward", action="store_true",
                    help="drop the reward from the encoder features and the decoder's"
                         " reward head")
    ap.add_argument("--no-observe-value", action="store_true",
                    help="COIN observes z alone (1-D) instead of (z, episodic return)")
    ap.add_argument("--ewc-max-tasks", type=int, default=8,
                    help="cap on accumulated EWC snapshots (cost grows linearly)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)

    import rl
    from rl import AmortisedCOINPPOAgent
    from baselines import fig3_common as f3
    from realtimecoin import RealTimeCOIN

    A, B = args.tasks
    global MC, FLAT, BLOCKS
    MC, FLAT = A, B                       # probe/trace slots reuse the 10c naming
    BLOCKS = ((A, args.rollouts), (B, args.rollouts))
    print(f"pair: {f3.TASK_NAMES[A]} -> {f3.TASK_NAMES[B]}, "
          f"{args.rollouts} rollouts each, seed {args.seed}", flush=True)

    t0 = time.perf_counter()
    experts = {t: pretrain_expert(t) for t in (MC, FLAT)}

    rl.seed_everything(args.seed)
    proto = f3.make_task_env(MC, None, 200)
    agent = AmortisedCOINPPOAgent(
        proto, CTX_IDS, prior_sd=0.5, kl_coef=KL_BETA,
        encoder_lr=3e-4, replay_capacity=512,
        value_coef=args.value_coef,
        encoder_reward=not args.no_encoder_reward,
        observe_value=not args.no_observe_value,
        ewc_coef=args.ewc_coef, ewc_head_coef=args.ewc_heads_coef,
        learn_gate=args.learn_gate,
        learn_gate_thresh=args.learn_gate_thresh, act_gate=args.act_gate,
        anchor_coef=args.anchor_coef,
        same_task_rollout=True, **f3.PPO_KWARGS)
    proto.close()
    # COIN at its published priors: the PEARL bottleneck holds z in the model's
    # native envelope (prior_sd 0.5 -> codes ~ +-1 at 2 sigma), so no retention or
    # process-noise surgery is needed; motor noise is the sensorimotor default pair.
    coin_kw = {}
    if args.stationary_sd is not None:
        a0 = 0.9425                       # prior_mean_retention (package default)
        coin_kw["prior_precision_drift"] = 1.0 / ((1.0 - a0) ** 2
                                                  * args.stationary_sd ** 2)
    # The COIN model must match what the agent will hand it: 2-D (z, return) unless
    # the value observation is ablated away, in which case it observes z alone.
    coin = RealTimeCOIN(rng=args.seed, sigma_motor_noise=0.0182,
                        state_dim=1 if args.no_observe_value else 2,
                        max_contexts=10, **coin_kw)

    probes = {t: collect_probe(agent, experts[t][2], t) for t in (MC, FLAT)}

    def encode(t):
        with torch.no_grad():
            m, s = agent.encoder.prefix_posterior(probes[t].to(agent.device),
                                                  SEG_STEPS)
        return float(m[:, -1].mean()), float(m[:, -1].std())

    n_total = sum(n for _, n in BLOCKS)
    code = np.full((n_total + 1, 2), np.nan)
    spread = np.full((n_total + 1, 2), np.nan)
    kpost = np.full(n_total, -1)
    vsteps = np.full(n_total, np.nan)
    W = agent.num_contexts
    pi_hist = np.full((n_total, W), np.nan)     # predicted pi (pre-evidence)
    rho_hist = np.full((n_total, W), np.nan)    # post-observation responsibilities
    wact_hist = np.full((n_total, W), np.nan)   # mean ACTING weights
    injected = set()
    head_ret_mid = np.full(agent.num_contexts, np.nan)
    code[0, 0], spread[0, 0] = encode(MC)
    code[0, 1], spread[0, 1] = encode(FLAT)

    i = 0
    for task, n_roll in BLOCKS:
        for _ in range(n_roll):
            envs = f3.make_task_envs(task, 8, seed=i * 100 + args.seed,
                                     max_episode_steps=200)
            r = agent.train_step(envs, coin, seg_steps=SEG_STEPS, mini_epochs=10,
                                 mb_size=64, enc_steps=160, mb_segments=4,
                                 carry_state=False)
            f3.close_envs(envs)
            # Diagnostic bootstrap: a newly instantiated head gets this block's
            # expert weights (in-place, so the head optimiser stays valid).
            for cid in (agent.context_keys[:-1] if not args.no_inject else []):
                if agent.context_init.get(cid, 0) == 1 and cid not in injected:
                    p_sd, v_sd, _ = experts[task]
                    agent.nets[cid][1].load_state_dict(p_sd)
                    agent.nets[cid][2].load_state_dict(v_sd)
                    injected.add(cid)
                    print(f"[inject] rollout {i}: head {cid} <- "
                          f"{f3.TASK_NAMES[task]} expert", flush=True)
            code[i + 1, 0], spread[i + 1, 0] = encode(MC)
            code[i + 1, 1], spread[i + 1, 1] = encode(FLAT)
            if not np.isfinite(code[i + 1]).all():
                bad = [n for n, q in agent.encoder.named_parameters()
                       if not torch.isfinite(q).all()]
                print(f"[NAN] first at rollout {i}: params {bad[:4]}; "
                      f"dyn={r['dyn_loss']:.5f} vloss={r['enc_value_loss']}",
                      flush=True)
            # End of block A: probe every head on task A BEFORE task B arrives. This
            # is what separates "block B destroyed the head" from "block A never
            # produced an individually competent head in the first place" -- the
            # end-of-run oracle probe cannot tell those apart.
            if args.probe_switch and i == BLOCKS[0][1] - 1:
                env_a = f3.make_task_env(MC, None, 200, train=False)
                mid = {j: eval_fixed_head(agent, env_a, j)
                       for j, cid in enumerate(agent.context_keys[:-1])
                       if agent.context_init.get(cid, 0) == 1}
                env_a.close()
                for j, v in mid.items():
                    head_ret_mid[j] = v
                print(f"[mid] end of block A, task A per head: "
                      f"{ {k: round(v) for k, v in mid.items()} }", flush=True)

            kpost[i] = int(coin.context_alignment()["K"])
            # Consolidate on a BIRTH: COIN announcing a new context is the model's own
            # statement that the task changed, so it is the boundary EWC needs without
            # the agent being told the schedule. The pool is a reservoir over the whole
            # stream, so the Fisher here covers everything seen so far.
            # Consolidate the HEADS on a birth too: at that moment the previous
            # context's head is the best it will ever be, and the arriving task is
            # about to compete for it.
            if (args.ewc_heads_coef > 0.0 and i > 0 and kpost[i] > kpost[i - 1]
                    and len(agent.ewc_head_tasks) < args.ewc_max_tasks
                    and len(agent.replay) > 0):
                hi = agent.consolidate_heads(SEG_STEPS, n_batches=16, mb_segments=4)
                print(f"[ewc-heads] rollout {i}: K {kpost[i-1]}->{kpost[i]}, "
                      f"consolidated (task {int(hi['n_head_tasks'])}, "
                      f"fisher trace {hi['head_fisher_trace']:.3e})", flush=True)

            if (args.ewc_coef > 0.0 and i > 0 and kpost[i] > kpost[i - 1]
                    and len(agent.ewc_tasks) < args.ewc_max_tasks
                    and len(agent.replay) > 0):
                info = agent.consolidate_encoder(SEG_STEPS, n_batches=32,
                                                 mb_segments=4)
                print(f"[ewc] rollout {i}: K {kpost[i-1]}->{kpost[i]}, "
                      f"consolidated (task {int(info['n_tasks'])}, "
                      f"fisher trace {info['fisher_trace']:.3e})", flush=True)
            vsteps[i] = r.get("ep_value_steps", np.nan)
            pi_hist[i] = np.nanmean(r["pi"], axis=0)
            rho_hist[i] = np.nanmean(np.asarray(r["rho"], dtype=float), axis=0)
            wact_hist[i] = np.nanmean(r["w_mean"], axis=0)
            if i % 10 == 0 or (task == FLAT and i - BLOCKS[0][1] < 6):
                gap = abs(code[i + 1, 0] - code[i + 1, 1])
                dom = int(np.nanargmax(wact_hist[i]))
                print(f"[{i:>3}] task={f3.TASK_NAMES[task]:<15} "
                      f"z_mc={code[i + 1, 0]:+.3f} z_flat={code[i + 1, 1]:+.3f} "
                      f"gap={gap:.3f} K={kpost[i]} vsteps={vsteps[i]:.0f} "
                      f"ret={np.nanmean(r['mean_episode_return']):.1f} "
                      f"anc={r.get('anchor_loss', float('nan')):.3f} "
                      f"act={dom}:{np.nanmax(wact_hist[i]):.2f} "
                      f"pi={np.round(np.nan_to_num(pi_hist[i][:6]), 2)} "
                      f"rho={np.round(np.nan_to_num(rho_hist[i][:6]), 2)}",
                      flush=True)
            i += 1

    # ---- the real z-marginal eval: does identification separate the pair? ----
    eval_ret, eval_head, eval_w = {}, {}, {}
    for t in (MC, FLAT):
        env = f3.make_task_env(t, None, 200, train=False)
        out = agent.evaluate_identifying(env, coin, n_episodes=EVAL_EPISODES,
                                         max_steps=200, seed=9000 + t)
        env.close()
        wm = np.nanmean(out["w_mean"], axis=0)
        eval_ret[t] = float(np.mean(out["returns"]))
        eval_head[t] = int(np.nanargmax(wm))
        eval_w[t] = float(np.nanmax(wm))
        print(f"[eval] {f3.TASK_NAMES[t]:<16}: return {eval_ret[t]:7.1f}  "
              f"head {eval_head[t]} (w={eval_w[t]:.2f})", flush=True)

    # ---- oracle routing: does ANY head still know each task? ----
    head_ret = np.full((2, agent.num_contexts), np.nan)
    for r, t in enumerate((MC, FLAT)):
        env = f3.make_task_env(t, None, 200, train=False)
        per = {}
        for j, cid in enumerate(agent.context_keys[:-1]):
            if agent.context_init.get(cid, 0) == 1:
                per[j] = eval_fixed_head(agent, env, j)
                head_ret[r, j] = per[j]
        env.close()
        top = max(per, key=per.get) if per else -1
        print(f"[oracle] {f3.TASK_NAMES[t]:<16}: best head {top} -> "
              f"{per.get(top, float('nan')):7.1f}   all: "
              f"{ {k: round(v) for k, v in per.items()} }", flush=True)

    np.savez(args.out, code=code, spread=spread, kpost=kpost, vsteps=vsteps,
             pi_hist=pi_hist, rho_hist=rho_hist, wact_hist=wact_hist,
             blocks=np.array(BLOCKS), head_ret=head_ret,
             head_ret_mid=head_ret_mid,
             eval_returns=np.array([eval_ret[MC], eval_ret[FLAT]]),
             eval_heads=np.array([eval_head[MC], eval_head[FLAT]]),
             eval_wmax=np.array([eval_w[MC], eval_w[FLAT]]),
             seconds=time.perf_counter() - t0)
    sw = BLOCKS[0][1]
    gap_start = abs(code[sw, 0] - code[sw, 1])
    gap_end = abs(code[-1, 0] - code[-1, 1])
    pooled = float(np.sqrt(np.nanmean(spread[-1] ** 2)))
    print(f"\ngap at FlatMC arrival: {gap_start:.3f}   gap at end: {gap_end:.3f}   "
          f"pooled probe sd: {pooled:.3f}   d-prime end: {gap_end / max(pooled, 1e-6):.1f}")
    print(f"K final: {kpost[-1]}   value steps/rollout (FlatMC block): "
          f"{np.nanmean(vsteps[sw:]):.1f}")
    print("PASS" if gap_end > max(3 * pooled, gap_start + 0.2) else "FAIL",
          "- divergence gate: gap_end > max(3*pooled_sd, gap_start+0.2)")
    print("PASS" if eval_head[MC] != eval_head[FLAT] else "FAIL",
          "- eval-separation gate: distinct dominant heads under z-marginal eval")
    # The gate the other two miss: separation and birth can both look healthy while
    # task A's head is quietly destroyed by the block-B stream.
    oracle_a = float(np.nanmax(head_ret[0]))
    print("PASS" if oracle_a > 140.0 else "FAIL",
          f"- retention gate: best head on task A = {oracle_a:.1f}"
          f" (self-identified {eval_ret[MC]:.1f}) -> "
          + ("ROUTING lost it (the head survives)" if oracle_a > 140.0
             else "the HEAD was overwritten"))


if __name__ == "__main__":
    main()

