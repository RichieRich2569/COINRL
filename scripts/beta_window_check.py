"""PEARL-bottleneck beta window: where between collapse and wander does kl_coef sit?

Encoder+decoder only (no COIN, no PPO): three-task joint stream (MC / CP / Mirror,
random-action segments with rewards -- the minimal stack's features), 150 iterations,
sweeping beta. Per beta, report:
  scale   -- population sd of the three task codes (want O(prior_sd)=0.5, neither
             exploding past ~3x nor shrinking to ~0)
  sep     -- min pairwise task gap / pooled within-task sd (informative latent)
  post_sd -- mean posterior sd vs prior_sd (== prior everywhere -> collapse)
Window = betas with sep comfortably > 1 and scale contained.

Usage: python beta_window_check.py
"""
import sys

REPO = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)

import numpy as np
import torch

import rl
from rl import AmortisedCOINPPOAgent
from baselines import fig3_common as f3

TASKS = (0, 3, 4)
BETAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
N_ITERS, SEG_STEPS, ENC_STEPS, MB = 150, 256, 160, 4


def segments(agent, task, n, seed):
    envs = f3.make_task_envs(task, n, None, 200)
    rl.seed_envs(envs, seed)
    out = []
    with torch.no_grad():
        for env in envs:
            obs_t = agent._flatten_obs(env.reset()[0])
            obs, act, nxt, rew = [], [], [], []
            for _ in range(SEG_STEPS):
                a = env.action_space.sample()
                next_obs, r, done, trunc, info = env.step(a)
                obs.append(obs_t.cpu())
                act.append(a)
                nxt.append(agent._flatten_obs(next_obs).cpu())
                rew.append(float(info.get("raw_reward", r))
                           if isinstance(info, dict) else float(r))
                if done or trunc:
                    next_obs, _ = env.reset()
                obs_t = agent._flatten_obs(next_obs)
            out.append(agent._segment_features(obs, act, nxt, rew=rew))
            env.close()
    return torch.cat(out)


def run(beta, seed=0):
    torch.set_num_threads(2)
    rl.seed_everything(seed)
    proto = f3.make_task_env(3, None, 200)
    agent = AmortisedCOINPPOAgent(proto, [0, 1, 2, "novel"], prior_sd=0.5,
                                  kl_coef=float(beta), encoder_lr=3e-4,
                                  replay_capacity=512, encoder_reward=True,
                                  **f3.PPO_KWARGS)
    proto.close()
    probes = {t: segments(agent, t, 8, 700 + t) for t in TASKS}
    for i in range(N_ITERS):
        for j, t in enumerate(TASKS):
            feats = segments(agent, t, 1, seed * 9999 + i * 10 + j)
            agent.replay.push(feats)
        agent._update_encoder(SEG_STEPS, ENC_STEPS, MB)
    mus, sds = [], []
    for t in TASKS:
        with torch.no_grad():
            m, s = agent.encoder.prefix_posterior(probes[t], SEG_STEPS)
        mus.append(float(m[:, -1].mean()))
        sds.append((float(m[:, -1].std()), float(s[:, -1].mean())))
    mus = np.array(mus)
    within = np.sqrt(np.mean([w ** 2 + p ** 2 for w, p in sds]))
    gaps = [abs(mus[a] - mus[b]) for a in range(3) for b in range(a + 1, 3)]
    return {"beta": beta, "scale": float(mus.std()),
            "sep": float(min(gaps) / max(within, 1e-8)),
            "post_sd": float(np.mean([p for _, p in sds])),
            "mus": mus.round(3)}


def main():
    import multiprocess as mp
    with mp.Pool(processes=len(BETAS)) as pool:
        res = [p.get() for p in [pool.apply_async(run, (b,)) for b in BETAS]]
    print(f"{'beta':>8} {'scale(sd of mus)':>17} {'min sep':>8} {'post_sd':>8}   mus")
    for r in res:
        print(f"{r['beta']:>8g} {r['scale']:>17.3f} {r['sep']:>8.1f} "
              f"{r['post_sd']:>8.3f}   {r['mus']}")
    print("\nprior_sd = 0.5. Window: sep >> 1, scale ~ 0.1-1.5, post_sd < prior.")


if __name__ == "__main__":
    main()

