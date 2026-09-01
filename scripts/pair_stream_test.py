"""Parametrised miniature pair-stream test: task A block then task B block, 2-D
stack + episodic value steps (stationary pi), expert-bootstrapped heads, per-rollout
probe-code traces, and a post-stream evaluate_identifying (the real z-marginal eval)
on both tasks. NOT a pilot -- a targeted probe of birth / z-divergence / eval routing
for one task pair. Set FIG3_TASK4=gravflip in the environment to make task 4 the
original InvertedCartPole.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--rollouts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0,
                    help="stream seed: seed_everything, COIN rng, rollout env seeds"
                         " (experts and probe/eval seeds stay fixed across arms)")
    ap.add_argument("--value-pi-source", default="stationary",
                    choices=["stationary", "predicted"])
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
          f"{args.rollouts} rollouts each, seed {args.seed}, "
          f"value_pi_source={args.value_pi_source}", flush=True)

    t0 = time.perf_counter()
    experts = {t: pretrain_expert(t) for t in (MC, FLAT)}

    rl.seed_everything(args.seed)
    proto = f3.make_task_env(MC, None, 200)
    agent = AmortisedCOINPPOAgent(
        proto, CTX_IDS, z_scale=2.0, prior_sd=0.5, kl_coef=0.0,
        encoder_lr=1.5e-4, replay_capacity=512,
        anchor_coef=0.0, anchor_window=64, anchor_warmup=0,
        rail_coef=1.0, z_channel_noise=0.4,
        value_coef=1e-3, decoder_residual=True, encoder_reward=True,
        observe_value=True, episodic_value_steps=True,
        value_pi_source=args.value_pi_source,
        same_task_rollout=True, **f3.PPO_KWARGS)
    proto.close()
    coin = RealTimeCOIN(rng=args.seed, sigma_motor_noise=0.05,
                        prior_mean_retention=0.9995,
                        state_dim=2,
                        process_noise_covariance=np.diag([0.0089 ** 2, 0.01 ** 2]),
                        max_contexts=10)

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
    injected = set()
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
            for cid in agent.context_keys[:-1]:
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
                      f"dyn={r['dyn_loss']:.5f} vloss={r['enc_value_loss']} "
                      f"vsteps={r['ep_value_steps']}", flush=True)
            kpost[i] = int(coin.context_alignment()["K"])
            vsteps[i] = r.get("ep_value_steps", np.nan)
            if i % 10 == 0:
                gap = abs(code[i + 1, 0] - code[i + 1, 1])
                print(f"[{i:>3}] task={f3.TASK_NAMES[task]:<15} "
                      f"z_mc={code[i + 1, 0]:+.3f} z_flat={code[i + 1, 1]:+.3f} "
                      f"gap={gap:.3f} K={kpost[i]} vsteps={vsteps[i]:.0f} "
                      f"ret={np.nanmean(r['mean_episode_return']):.1f}", flush=True)
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

    np.savez(args.out, code=code, spread=spread, kpost=kpost, vsteps=vsteps,
             blocks=np.array(BLOCKS),
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


if __name__ == "__main__":
    main()

