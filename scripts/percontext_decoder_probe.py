"""B4: does LLIRL's per-cluster dynamics model identify a task better than our
shared z-conditioned decoder?

Our task identity lives in a COORDINATE -- a scalar z read by one decoder -- and a
coordinate has a gauge, which is what every drift failure so far has come from (codes
translating together, COIN's stored centres going stale, an arriving task settling on a
vacated coordinate). LLIRL instead gives each cluster its OWN dynamics model and infers
identity from which model explains the data best. There is then no shared coordinate to
translate.

This probe is offline: no policy, no COIN, no RL. Fit both schemes on the same segments
and ask which one separates the two tasks more sharply, on its own terms.

  shared    encoder q(z|c) + one decoder f(s, a, z) -> s'   (the current architecture)
            separability = d' between the two tasks' codes
  per-task  one decoder f_k(s, a) -> s' per task            (LLIRL)
            separability = d' of the per-segment log-likelihood MARGIN between models

Both are reported as d' on held-out segments, so they are directly comparable: how many
noise standard deviations apart are the two tasks under each scheme.

Usage: python percontext_decoder_probe.py [--steps 600]
"""
import argparse
import sys
import time

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

import numpy as np

SEG_LEN = 256
N_TRAIN, N_PROBE = 16, 8
CTX_IDS = [0, 1, "novel"]


def collect(agent, env_fn, n_seg, seed):
    import torch

    feats = []
    for i in range(n_seg):
        env = env_fn()
        obs_t = agent._flatten_obs(env.reset(seed=seed + i)[0])
        env.action_space.seed(seed + i)
        obs, act, nxt, rew = [], [], [], []
        for _ in range(SEG_LEN):
            a = env.action_space.sample()
            next_obs, r, done, trunc, _ = env.step(a)
            obs.append(obs_t.cpu())
            act.append(a)
            nxt.append(agent._flatten_obs(next_obs).cpu())
            rew.append(float(r))
            if done or trunc:
                next_obs, _ = env.reset()
            obs_t = agent._flatten_obs(next_obs)
        feats.append(agent._segment_features(obs, act, nxt, rew=rew))
        env.close()
    return feats


def dprime(a, b):
    """Separation of two sample sets in pooled-sd units."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float(abs(a.mean() - b.mean()) / max(pooled, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.set_num_threads(3)

    import rl
    from rl import AmortisedCOINPPOAgent, _MLP
    from environments import CartPoleXEnv

    rl.seed_everything(args.seed)
    proto = CartPoleXEnv(gravity=9.8, max_episode_steps=200)
    agent = AmortisedCOINPPOAgent(proto, CTX_IDS, prior_sd=0.5, kl_coef=1e-4,
                                  encoder_lr=3e-4, replay_capacity=512,
                                  encoder_reward=True, device="cpu")
    proto.close()

    env_a = lambda: CartPoleXEnv(gravity=9.8, max_episode_steps=200)
    env_b = lambda: CartPoleXEnv(force_mag=-10.0, max_episode_steps=200)
    t0 = time.perf_counter()
    tr_a = collect(agent, env_a, N_TRAIN, 100)
    tr_b = collect(agent, env_b, N_TRAIN, 200)
    pr_a = collect(agent, env_a, N_PROBE, 900)
    pr_b = collect(agent, env_b, N_PROBE, 950)
    print(f"collected ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---------------- scheme 1: shared z-conditioned decoder ----------------
    for f in tr_a + tr_b:
        agent.replay.push(f)
    agent._update_encoder(SEG_LEN, args.steps, 4)
    with torch.no_grad():
        za = agent.encoder.prefix_posterior(torch.cat(pr_a), SEG_LEN)[0][:, -1].numpy()
        zb = agent.encoder.prefix_posterior(torch.cat(pr_b), SEG_LEN)[0][:, -1].numpy()
    d_shared = dprime(za, zb)

    # ---------------- scheme 2: one decoder per task (LLIRL) ----------------
    obs_dim, act_dim = agent.obs_dim, agent.act_dim
    sa = slice(0, obs_dim + act_dim)

    def fit(segs, seed):
        torch.manual_seed(seed)
        net = _MLP(obs_dim + act_dim, obs_dim, 64)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4)
        data = torch.cat(segs)
        for _ in range(args.steps):
            idx = torch.randperm(data.shape[0])[:1024]
            b = data[idx]
            loss = (net(b[:, sa]) - b[:, -obs_dim:]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return net

    f_a, f_b = fit(tr_a, 1), fit(tr_b, 2)

    def margin(segs):
        """Per-segment mean log-likelihood margin log p(D|f_A) - log p(D|f_B).

        Fixed-variance Gaussian observation model, as LLIRL uses, so the margin is
        (up to a constant) the difference of the two models' squared errors.
        """
        out = []
        with torch.no_grad():
            for s in segs:
                ea = (f_a(s[:, sa]) - s[:, -obs_dim:]).pow(2).mean()
                eb = (f_b(s[:, sa]) - s[:, -obs_dim:]).pow(2).mean()
                out.append(float(eb - ea))      # >0 => f_A explains it better
        return out

    ma, mb = margin(pr_a), margin(pr_b)
    d_percontext = dprime(ma, mb)

    print(f"\n{'scheme':<24} {'task A':>12} {'task B':>12} {'d-prime':>9}")
    print(f"{'shared z + 1 decoder':<24} {za.mean():12.4f} {zb.mean():12.4f} "
          f"{d_shared:9.1f}")
    print(f"{'per-task decoders':<24} {np.mean(ma):12.4f} {np.mean(mb):12.4f} "
          f"{d_percontext:9.1f}")
    print(f"\nshared: task means are z codes; per-task: mean LL margin "
          f"(sign = which model wins)")
    print(f"correct sign on both probes: "
          f"{bool(np.mean(ma) > 0 and np.mean(mb) < 0)}")
    print(f"\nd-prime ratio (per-task / shared): "
          f"{d_percontext / max(d_shared, 1e-9):.2f}"
          f"   >1 means the LLIRL scheme separates these tasks more sharply")
    print(f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
