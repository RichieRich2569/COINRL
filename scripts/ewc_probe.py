"""Does EWC replace the replay pool for keeping the encoder trained?

Focused, PPO-free probe of the one thing the reservoir exists to prevent: on a
BLOCKED stream the encoder is fitted only on the current task, so it forgets the
previous one and its code drifts off it -- which is what makes COIN's stored context
centres stale.

Protocol (no policy, no returns, random actions -- the contingency is in the
transitions):

  phase 1   pool = task A only            -> train encoder
            measure z_A, z_B and the dynamics loss on HELD-OUT A segments
  consolidate (EWC arms only)
  phase 2   pool depends on the arm       -> train encoder again
            re-measure

Arms
  replay      pool = A + B      the current default (reservoir)
  none        pool = B only     no protection: the catastrophic-forgetting control
  ewc         pool = B only     + EWC penalty on the encoder

What is being measured
  forget    dynamics loss on held-out A segments, phase 2 / phase 1. 1.0 = perfect
            retention; large = the encoder can no longer model task A.
  drift     |z_A(after) - z_A(before)|, in units of the probe spread. This is the
            quantity that invalidates COIN's centres -- a task whose code MOVES makes
            every stored context stale even if nothing was forgotten.
  gap       |z_A - z_B| after phase 2: the codes must still be separable.
  d'        gap / pooled probe sd -- separability given the encoder's own noise.
  COIN      the measured codes replayed as a blocked stream of trials: does COIN end
            with two contexts, and does it give the two blocks different ones?

Usage: python ewc_probe.py [--ewc-coef 1e3 ...]
"""
import argparse
import sys
import time

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

import numpy as np

SEG_LEN = 256
N_SEG = 16           # segments per task for training
N_PROBE = 8          # held-out segments per task for measurement
CTX_IDS = [0, 1, 2, 3, "novel"]


def collect(agent, env_fn, n_seg, seed):
    """Random-action segments; the force-sign contingency shows up in transitions."""
    import torch

    feats = []
    for i in range(n_seg):
        env = env_fn()
        env.reset(seed=seed + i)
        env.action_space.seed(seed + i)
        obs_t = agent._flatten_obs(env.reset(seed=seed + i)[0])
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


def code_of(agent, probe):
    """(mean, sd) of the segment-final posterior over a set of probe segments."""
    import torch

    with torch.no_grad():
        m, _ = agent.encoder.prefix_posterior(torch.cat(probe).to(agent.device), SEG_LEN)
    z = m[:, -1].cpu().numpy()
    return float(z.mean()), float(z.std())


def dyn_on(agent, probe):
    """Dynamics loss on held-out segments -- the forgetting measure."""
    import torch

    with torch.no_grad():
        feats = torch.cat(probe).to(agent.device)
        mean, sd = agent.encoder.prefix_posterior(feats, SEG_LEN)
        z = mean[:, :-1].reshape(-1)
        return float(agent._dyn_loss(feats, z, SEG_LEN).item())


def build(seed, ewc_coef, protect_decoder=True):
    import torch

    import rl
    from rl import AmortisedCOINPPOAgent
    from environments import CartPoleXEnv

    rl.seed_everything(seed)
    proto = CartPoleXEnv(gravity=9.8, max_episode_steps=200)
    agent = AmortisedCOINPPOAgent(
        proto, CTX_IDS, prior_sd=0.5, kl_coef=1e-4, encoder_lr=3e-4,
        replay_capacity=512, encoder_reward=True, ewc_coef=ewc_coef,
        ewc_protect_decoder=protect_decoder, device="cpu")
    proto.close()
    return agent


def coin_check(z_a, sd_a, z_b, sd_b, seed=0, n=8):
    """Replay the measured codes as a blocked stream: does COIN separate them?"""
    from realtimecoin import RealTimeCOIN

    rng = np.random.default_rng(seed)
    coin = RealTimeCOIN(rng=seed, sigma_motor_noise=0.0182, max_contexts=10)
    dom = []
    for mu, sd in ((z_a, sd_a), (z_b, sd_b)):
        for _ in range(n):
            coin.observe_q(None)
            coin.sigma_sensory_noise = max(float(sd), 1e-3)
            coin.observe_y(float(rng.normal(mu, max(sd, 1e-6))))
            K = int(coin.context_alignment()["K"])
            rho = np.asarray(coin.responsibilities_vector(), dtype=float)[:K + 1]
            dom.append(int(np.nanargmax(rho)))
    return int(coin.context_alignment()["K"]), dom[n - 1], dom[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ewc-coef", type=float, nargs="+", default=[1e2, 1e4])
    ap.add_argument("--steps", type=int, default=400, help="encoder steps per phase")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)

    from environments import CartPoleXEnv

    env_a = lambda: CartPoleXEnv(gravity=9.8, max_episode_steps=200)
    env_b = lambda: CartPoleXEnv(force_mag=-10.0, max_episode_steps=200)

    t0 = time.perf_counter()
    # (label, ewc_coef, protect_decoder, keep_pool). keep_pool=True is the reservoir
    # arm (task A's data survives into phase 2); False is the blocked stream EWC has
    # to cope with on its own. "both" runs the two together.
    arms = [("replay", 0.0, True, True), ("none", 0.0, True, False)]
    arms += [(f"ewc {c:g}", c, True, False) for c in args.ewc_coef]
    arms += [(f"both {c:g}", c, True, True) for c in args.ewc_coef]
    arms += [(f"enc-only {c:g}", c, False, False) for c in args.ewc_coef[-1:]]
    rows = []

    for name, coef, protect_dec, keep_pool in arms:
        agent = build(args.seed, coef, protect_decoder=protect_dec)
        train_a = collect(agent, env_a, N_SEG, seed=100)
        train_b = collect(agent, env_b, N_SEG, seed=200)
        probe_a = collect(agent, env_a, N_PROBE, seed=900)
        probe_b = collect(agent, env_b, N_PROBE, seed=950)

        # ---- phase 1: task A only ----
        for f in train_a:
            agent.replay.push(f)
        agent._update_encoder(SEG_LEN, args.steps, 4)
        za1, sda1 = code_of(agent, probe_a)
        zb1, _ = code_of(agent, probe_b)
        dyn_a1 = dyn_on(agent, probe_a)

        if coef > 0.0:
            agent.consolidate_encoder(SEG_LEN, n_batches=32, mb_segments=4)

        # ---- phase 2: the arm's pool ----
        if not keep_pool:
            agent.replay.clear()            # blocked stream, no old data kept
        for f in train_b:
            agent.replay.push(f)
        stats2 = agent._update_encoder(SEG_LEN, args.steps, 4)

        za2, sda2 = code_of(agent, probe_a)
        zb2, sdb2 = code_of(agent, probe_b)
        dyn_a2 = dyn_on(agent, probe_a)

        pooled = float(np.sqrt(0.5 * (sda2 ** 2 + sdb2 ** 2)))
        gap = abs(za2 - zb2)
        drift = abs(za2 - za1)
        K, dom_a, dom_b = coin_check(za2, sda2, zb2, sdb2, seed=args.seed)
        rows.append((name, dyn_a1, dyn_a2, dyn_a2 / max(dyn_a1, 1e-12), za1, za2,
                     drift, gap, gap / max(pooled, 1e-9), K, dom_a, dom_b,
                     stats2.get("ewc_loss", float("nan"))))
        print(f"  {name:<12} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"\n{'arm':<12} {'dynA p1':>9} {'dynA p2':>9} {'forget':>7} "
          f"{'zA p1':>7} {'zA p2':>7} {'drift':>7} {'gap':>7} {'d-prime':>8} "
          f"{'K':>2} {'ctxA':>5} {'ctxB':>5} {'ewc_loss':>10}")
    for r in rows:
        print(f"{r[0]:<12} {r[1]:9.5f} {r[2]:9.5f} {r[3]:7.2f} "
              f"{r[4]:+7.3f} {r[5]:+7.3f} {r[6]:7.3f} {r[7]:7.3f} {r[8]:8.1f} "
              f"{r[9]:2d} {r[10]:5d} {r[11]:5d} {r[12]:10.3e}")
    print("\nforget = dynA(p2)/dynA(p1), 1.0 is perfect retention")
    print("drift  = |zA(p2) - zA(p1)|; this is what makes COIN's stored centres stale")
    print("ctxA/ctxB = COIN's dominant context at the end of each block "
          "(different = it separated them)")


if __name__ == "__main__":
    main()
