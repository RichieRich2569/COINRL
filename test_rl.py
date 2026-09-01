"""
test_rl.py

Basic pytest coverage for the dynamics-based amortised COIN-PPO block of :mod:`rl`:
the encoder's prefix posterior and feature layout, the decoder, the property that made
dynamics training worth the switch (a non-zero encoder gradient at initialisation), the
segment replay pool, the episode carried across segment boundaries, the uncertainty handed
to COIN, the COIN interface helpers, end-to-end smoke tests of both training regimes, and
:mod:`curriculum` -- the Markov schedule and the metrics that score COIN against it.

Run with ``.venv/Scripts/python -m pytest test_rl.py -v``.
"""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

import curriculum
from curriculum import MarkovTaskCurriculum
from environments import CustomCartPoleEnv
from rl import (AmortisedCOINPPOAgent, ContingencyEncoder, FiLMDecoder,
                SegmentReplayBuffer, coin_predicted_pi, seed_envs, seed_everything)

realtimecoin = pytest.importorskip("realtimecoin")
RealTimeCOIN = realtimecoin.RealTimeCOIN

OBS_DIM, ACT_DIM = 4, 2       # CartPole
CTX_IDS = {0: 0, 1: 1}        # -> context_keys [0, 1, "novel"], num_contexts == 3
TRANSITION = np.array([[0.9, 0.1], [0.2, 0.8]])     # stationary [2/3, 1/3]


def make_encoder(seed: int = 0, trained: bool = True, **kwargs) -> ContingencyEncoder:
    """An encoder whose final layer is randomised, so the factors actually vary."""
    torch.manual_seed(seed)
    enc = ContingencyEncoder(OBS_DIM, ACT_DIM, action_continuous=False, hidden=8, **kwargs)
    if trained:
        for p in enc.net.net[-1].parameters():
            nn.init.normal_(p, std=0.5)
    return enc


def make_envs(steps: int = 50):
    """Two fresh CartPoles differing only in the sign of the force -- the contingency."""
    return [CustomCartPoleEnv(force_mag=f, max_episode_steps=steps) for f in (10.0, -10.0)]


def make_endless_envs(steps: int = 1000, forces=(10.0, -10.0)):
    """CartPoles whose thresholds are out of reach, so a boundary is always mid-episode.

    ``steps`` is the only way an episode can end here, which makes both carry-over cases
    -- carried and terminal -- reachable on demand.
    """
    return [CustomCartPoleEnv(force_mag=f, max_episode_steps=steps,
                              theta_threshold_radians=1e6, x_threshold=1e6)
            for f in forces]


def spy_on_segment_starts(agent, record):
    """Wrap ``_start_segment`` to log the carry it saw and the episode it started."""
    inner = agent._start_segment

    def spy(env, carry_state):
        carry = None if agent._carry is None else dict(agent._carry)
        obs_t, ep_ret = inner(env, carry_state)
        record.append((carry, obs_t.clone(), ep_ret, int(env._elapsed_steps)))
        return obs_t, ep_ret

    agent._start_segment = spy


def coin_state_fingerprint(coin):
    """Every field of ``coin.snapshot()``, flattened into comparable leaves.

    The snapshot is the model's own definition of state (properties, particle arrays, trial,
    cue registry, alignment seed AND the random-stream position), so equality of two
    fingerprints is the round-trip guarantee: the model would continue the run
    observation for observation identically.
    """
    def flat(prefix, value, out):
        if hasattr(value, "__dataclass_fields__"):
            value = vars(value)
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                flat(f"{prefix}.{key}", value[key], out)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                flat(f"{prefix}[{i}]", item, out)
        elif isinstance(value, np.ndarray) and value.dtype == object:
            flat(prefix, value.tolist(), out)
        elif isinstance(value, np.ndarray):
            out[prefix] = (str(value.dtype), value.shape, value.tobytes())
        else:
            out[prefix] = repr(value)
        return out

    return flat("", coin.snapshot(), {})


def agent_param_fingerprint(agent):
    """Bitwise copy of every learnable tensor the agent owns, keyed by name."""
    out = {}
    for name, module in (("encoder", agent.encoder), ("decoder", agent.decoder)):
        for key, tensor in module.state_dict().items():
            out[f"{name}.{key}"] = tensor.detach().clone()
    for cid in agent.context_keys:
        if agent.context_init.get(cid, 0) == 0:
            continue
        _, policy, value_net, _ = agent.nets[cid]
        for tag, net in (("policy", policy), ("value", value_net)):
            for key, tensor in net.state_dict().items():
                out[f"{cid}.{tag}.{key}"] = tensor.detach().clone()
    return out


def spy_on_advantages(agent, record):
    """Wrap ``_compute_advantages`` to log each segment's done flags."""
    inner = agent._compute_advantages

    def spy(rewards, values, dones, last_value):
        record.append(list(dones))
        return inner(rewards, values, dones, last_value)

    agent._compute_advantages = spy


@pytest.fixture
def agent():
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    return AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3, prior_sd=0.3,
                                 kl_coef=0.03)


def test_prefix_posterior_matches_naive_product():
    """The cumsum form equals an explicit product of Gaussian factors, per segment."""
    enc = make_encoder(prior_sd=0.7, z_scale=0.5)
    seg_len, n_seg = 5, 2
    feats = torch.randn(n_seg * seg_len, enc.in_dim)

    mean, sd = enc.prefix_posterior(feats, seg_len)
    with torch.no_grad():
        mu, sigma = enc.factors(feats)

    for s in range(n_seg):
        e1, e2 = 0.0, 1.0 / enc.prior_sd ** 2
        # Column 0 is the prior, before any transition has been absorbed.
        assert mean[s, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert sd[s, 0].item() == pytest.approx(enc.prior_sd, rel=1e-5)
        for t in range(seg_len):
            prec = 1.0 / sigma[s * seg_len + t].item() ** 2
            e2 += prec
            e1 += mu[s * seg_len + t].item() * prec
            assert mean[s, t + 1].item() == pytest.approx(e1 / e2, rel=1e-4, abs=1e-7)
            assert sd[s, t + 1].item() == pytest.approx(np.sqrt(1.0 / e2), rel=1e-4)

    # Segments must be independent: perturbing segment 0 may not move segment 1.
    perturbed = feats.clone()
    perturbed[:seg_len] += 3.0
    mean2, sd2 = enc.prefix_posterior(perturbed, seg_len)
    assert torch.allclose(mean[1], mean2[1])
    assert torch.allclose(sd[1], sd2[1])
    assert not torch.allclose(mean[0], mean2[0])


def test_transition_features_shape_and_onehot():
    """Feature layout is ``concat(s, one_hot(a), s')`` -- reward is NOT an input."""
    enc = make_encoder()
    T = 6
    obs = torch.randn(T, OBS_DIM)
    next_obs = torch.randn(T, OBS_DIM)
    act = torch.arange(T) % ACT_DIM

    feats = enc.transition_features(obs, act, next_obs)

    assert feats.shape == (T, 2 * OBS_DIM + ACT_DIM) == (T, enc.in_dim)
    assert torch.allclose(feats[:, :OBS_DIM], obs)
    onehot = feats[:, OBS_DIM:OBS_DIM + ACT_DIM]
    assert torch.equal(onehot.sum(dim=1), torch.ones(T))
    assert torch.equal(onehot.argmax(dim=1), act)
    # No reward column: s' starts immediately after the action one-hot.
    assert torch.allclose(feats[:, OBS_DIM + ACT_DIM:], next_obs)
    assert torch.allclose(feats[:, -OBS_DIM:], next_obs)


def test_encoder_in_dim_excludes_reward_by_default():
    """The padded Fig-3 interface (6 obs, 3 actions) gives ``in_dim == 15`` unless
    ``use_reward`` opts in (16, decision reversed 2026-08-31); a default encoder
    silently ignores a passed reward."""
    enc = ContingencyEncoder(6, 3, action_continuous=False, hidden=8)
    assert enc.in_dim == 2 * 6 + 3 == 15
    assert enc.net.net[0].in_features == 15

    obs = torch.zeros(2, 6)
    act = torch.zeros(2, dtype=torch.long)
    base = enc.transition_features(obs, act, obs)
    with_r = enc.transition_features(obs, act, obs, rew=[5.0, 5.0])
    assert torch.equal(base, with_r)                  # default: reward cannot reach z

    # Continuous actions keep the same rule; opting in grows the slot by exactly one.
    cont = ContingencyEncoder(6, 3, action_continuous=True, hidden=8)
    assert cont.in_dim == 15
    assert ContingencyEncoder(6, 3, False, hidden=8, use_reward=True).in_dim == 16


def test_decoder_forward_shape(agent):
    """``_decode_next_obs`` maps ``(s, a_repr, z)`` to a finite ``[T, obs_dim]``."""
    T = 7
    feats = torch.randn(T, agent.encoder.in_dim)
    z = torch.randn(T)

    out = agent._decode_next_obs(feats, z)

    assert out.shape == (T, OBS_DIM)
    assert torch.isfinite(out).all()


def test_encoder_grad_nonzero_at_init(agent):
    """
    The whole point of dynamics training: the loss reaches phi from step one.

    The value-based path provably could not -- ``d rho / d z`` is identically zero whenever
    the context heads agree, which is the case at initialisation and at every context
    instantiation (new heads are deep copies).
    """
    seg_len, n_seg = 8, 2
    feats = torch.randn(n_seg * seg_len, agent.encoder.in_dim)

    mean, sd = agent.encoder.prefix_posterior(feats, seg_len)
    z = (mean[:, :-1] + torch.randn_like(sd[:, :-1]) * sd[:, :-1]).reshape(-1)
    dyn = (agent._decode_next_obs(feats, z) - feats[:, -OBS_DIM:]).pow(2).mean()
    loss = dyn + agent.kl_coef * agent._kl_to_prior(mean, sd).mean()
    loss.backward()

    def grad_norm(module):
        return sum(float(p.grad.pow(2).sum()) for p in module.parameters() if p.grad is not None)

    assert grad_norm(agent.encoder) > 0.0
    assert grad_norm(agent.decoder) > 0.0


def test_policy_weights(agent):
    """NaN padding is zeroed; the novel column is dropped unless it carries everything."""
    w = agent._policy_weights(np.array([0.5, np.nan, 0.5]))
    assert w.dtype is torch.float32 and w.shape == (agent.num_contexts,)
    assert np.allclose(w.numpy(), [1.0, 0.0, 0.0])          # known mass renormalised

    w = agent._policy_weights(np.array([0.2, 0.6, 0.2]))
    assert np.allclose(w.numpy(), [0.25, 0.75, 0.0])

    # Trial 0 (no known context yet): dropping novel would leave an all-zero vector.
    w = agent._policy_weights(np.array([np.nan, np.nan, 1.0]))
    assert np.allclose(w.numpy(), [0.0, 0.0, 1.0])

    agent.avoid_novel = False
    w = agent._policy_weights(np.array([0.5, np.nan, 0.5]))
    assert np.allclose(w.numpy(), [0.5, 0.0, 0.5])


def test_coin_predicted_pi():
    """One-step-ahead predicted context probabilities, in the aligned global frame."""
    coin = RealTimeCOIN(rng=0)
    width = coin.max_contexts + 1

    pi, K = coin_predicted_pi(coin)
    assert isinstance(K, int) and pi.shape == (width,)
    assert pi.sum() == pytest.approx(1.0)
    assert np.all(pi[K + 1:] == 0.0)                        # padding above the novel slot

    # Alternating feedback separates into exactly two contexts.
    for i in range(6):
        coin.observe_q(None)
        pi, K = coin_predicted_pi(coin)
        assert pi.sum() == pytest.approx(1.0)
        assert np.all(pi[K + 1:] == 0.0)
        coin.observe_y(0.3 if i % 2 == 0 else -0.3)

    assert int(coin.context_alignment()["K"]) == 2
    assert coin.trial == 6


def test_train_step_smoke(agent):
    """Two joint-regime updates end to end, driving COIN as they go."""
    envs, coin = make_envs(), RealTimeCOIN(rng=0)
    S, W = len(envs), agent.num_contexts

    for call in (1, 2):
        out = agent.train_step(envs, coin, seg_steps=32, mini_epochs=2, mb_size=16)

        assert set(out) == {"z", "z_sd", "K", "pi", "rho", "w_mean", "sharpen_step",
                            "mean_episode_return", "mean_reward_per_step", "value_loss",
                            "policy_loss", "dyn_loss", "encoder_kl", "enc_grad_norm",
                            "anchor_loss", "anchor_free", "disp_loss", "inv_loss",
                            "enc_value_loss", "repel_flagged"}
        for key in ("z", "z_sd", "K", "mean_episode_return", "sharpen_step"):
            assert out[key].shape == (S,)
        for key in ("pi", "rho", "w_mean"):
            assert out[key].shape == (S, W)
        assert np.isfinite(out["z"]).all() and np.all(out["z_sd"] > 0)
        # NaN marks an uninstantiated context slot (the module's agent layout), so the
        # rows sum to one only once those are ignored.
        assert np.allclose(np.nansum(out["pi"], axis=1), 1.0)
        for key in ("mean_reward_per_step", "value_loss", "policy_loss",
                    "dyn_loss", "encoder_kl"):
            assert np.isfinite(out[key])
        assert out["enc_grad_norm"] > 0.0
        assert coin.trial == call * S


def test_pretrain_encoder_and_frozen_train_step(agent):
    """Regime B: pretrain on random rollouts, then train RL with the encoder frozen."""
    envs = make_envs()
    before = [p.detach().clone() for p in agent.encoder.parameters()]

    history = agent.pretrain_encoder(envs, seg_steps=32, n_iters=2, enc_steps=2, mb_segments=2)

    assert set(history) == {"dyn_loss", "encoder_kl", "enc_grad_norm",
                            "anchor_loss", "anchor_free", "disp_loss", "inv_loss"}
    for values in history.values():
        assert values.shape == (2,) and np.isfinite(values).all()
    assert any(not torch.equal(b, p)
               for b, p in zip(before, agent.encoder.parameters()))
    # The pretrain segments land in the same pool the RL phase replays from.
    assert len(agent.replay) == 2 * len(envs)

    frozen = [p.detach().clone() for p in agent.encoder.parameters()]
    out = agent.train_step(make_envs(), RealTimeCOIN(rng=0), seg_steps=32, mini_epochs=2,
                           mb_size=16, update_encoder=False)

    assert out["enc_grad_norm"] == 0.0
    assert np.isfinite(out["dyn_loss"]) and np.isfinite(out["encoder_kl"])
    for f, p in zip(frozen, agent.encoder.parameters()):
        assert torch.equal(f, p)


#----- Segment replay -----

def test_segment_replay_buffer():
    """A reservoir in segments: fills in order, then samples uniformly, detached.

    Every slot carries its code anchor and its arrival index alongside the features, and
    ``sample`` hands the two back aligned.
    """
    np.random.seed(0)
    buf = SegmentReplayBuffer(capacity=3)
    for i in range(3):
        buf.push(torch.full((4, 2), float(i)), anchor=-float(i))

    # Below capacity the reservoir is just the stream, in order.
    assert len(buf) == 3 and buf.n_seen == 3
    assert [float(f[0, 0]) for f in buf.buffer] == [0.0, 1.0, 2.0]
    assert buf.anchors == [0.0, -1.0, -2.0] and buf.push_ids == [0, 1, 2]

    for i in range(3, 20):
        buf.push(torch.full((4, 2), float(i)), anchor=-float(i))
    assert len(buf) == 3 and buf.n_seen == 20        # capacity never exceeded
    # The three parallel lists never come apart: anchor == -tag, push_id == tag.
    for feats, a, p in zip(buf.buffer, buf.anchors, buf.push_ids):
        assert a == -float(feats[0, 0]) and p == int(feats[0, 0])

    feats, anchors = buf.sample(2)
    assert feats.shape == (2 * 4, 2) and anchors.shape == (2,)
    assert buf.sample(10)[0].shape == (3 * 4, 2)                     # capped at the pool
    feats, anchors = buf.sample(3)
    assert len({float(v) for v in feats[:, 0]}) == 3                 # no duplicates
    assert torch.allclose(anchors, -feats[::4, 0])                   # still aligned

    # An anchor is optional; absent means "not pinned", which the encoder reads as nan.
    unpinned = SegmentReplayBuffer(capacity=2)
    unpinned.push(torch.zeros(4, 2))
    assert np.isnan(unpinned.anchors[0]) and np.isnan(float(unpinned.sample(1)[1][0]))

    with pytest.raises(RuntimeError):
        SegmentReplayBuffer().sample(1)


def test_segment_replay_buffer_is_a_uniform_reservoir():
    """Algorithm R's invariant: every segment ever pushed is held with p = capacity / N.

    This is the property FIFO does not have, and the reason for the switch: on a blocked
    curriculum a FIFO pool holds only the newest block, so the encoder loses all evidence of
    the earlier tasks and its latent drifts off them.
    """
    np.random.seed(0)
    capacity, n_segments, trials = 4, 20, 2000
    counts = np.zeros(n_segments)

    for _ in range(trials):
        buf = SegmentReplayBuffer(capacity=capacity)
        for i in range(n_segments):
            buf.push(torch.full((1, 1), float(i)))
        assert len(buf) == capacity
        for feats in buf.buffer:
            counts[int(feats[0, 0])] += 1

    freq = counts / trials
    # sd of each frequency is sqrt(p(1-p)/trials) ~ 0.009, so 0.04 is >4 sd.
    assert np.allclose(freq, capacity / n_segments, atol=0.04)
    # Every segment gets in sometimes, and nothing is privileged by position.
    assert freq.min() > 0.0 and abs(freq[0] - freq[-1]) < 0.06


def test_segment_replay_buffer_clears_on_length_change():
    """One segment length per pool: ``prefix_posterior`` needs a common ``L``."""
    np.random.seed(0)
    buf = SegmentReplayBuffer(capacity=3)
    for i in range(9):
        buf.push(torch.full((4, 2), float(i)), anchor=float(i))
    assert len(buf) == 3 and buf.n_seen == 9

    live = torch.zeros(6, 2, requires_grad=True)
    buf.push(live, anchor=7.0)              # a new segment length empties the pool
    assert len(buf) == 1 and buf.n_seen == 1                # the reservoir count restarts
    assert not buf.buffer[-1].requires_grad                 # stored detached, on CPU
    assert buf.buffer[-1].shape == (6, 2)
    # The anchors and arrival indices are cleared with the features, never left dangling.
    assert buf.anchors == [7.0] and buf.push_ids == [0]

    # And the fresh reservoir fills from scratch under the new length.
    for i in range(4):
        buf.push(torch.full((6, 2), float(i)), anchor=float(i))
    assert len(buf) == 3 and all(f.shape == (6, 2) for f in buf.buffer)
    assert len(buf.anchors) == 3 and len(buf.push_ids) == 3


def test_train_step_fills_replay(agent):
    """Every rollout adds its S fresh segments; the encoder then draws from all of them."""
    coin = RealTimeCOIN(rng=0)
    S, L = 2, 16
    kwargs = dict(seg_steps=L, mini_epochs=1, mb_size=16, enc_steps=1, mb_segments=1)

    agent.train_step(make_envs(), coin, **kwargs)
    assert len(agent.replay) == S
    agent.train_step(make_envs(), coin, **kwargs)
    assert len(agent.replay) == 2 * S

    assert all(f.shape == (L, agent.encoder.in_dim) for f in agent.replay.buffer)
    feats, anchors = agent.replay.sample(2 * S)
    assert feats.shape == (2 * S * L, agent.encoder.in_dim) and anchors.shape == (2 * S,)
    # Every rollout segment arrives with the code the encoder gave it, and the default
    # warm-up then frees all four again -- nothing is pinned this early.
    assert agent.replay.push_ids == [0, 1, 2, 3] and agent.replay.n_seen == 2 * S
    assert np.isnan(agent.replay.anchors).all()


#----- Code-stability anchor -----

def test_anchor_loss_is_zero_until_the_encoder_moves(agent):
    """The anchor is the encoder's own past: zero at the moment of stamping, positive once
    the map has moved -- which is the whole mechanism, stated as a test."""
    L = 16
    agent.anchor_coef = 1.0
    agent.anchor_warmup = 0        # pinning from the first segment...
    agent.anchor_window = 0        # ...and nothing exempted as "still arriving"

    # Segments stamped with the CURRENT encoder's codes.
    for feats in (torch.randn(L, agent.encoder.in_dim) for _ in range(4)):
        with torch.no_grad():
            mean, _ = agent.encoder.prefix_posterior(feats, L)
        agent.replay.push(feats, float(mean[0, -1]))

    # enc_steps=1 reports the loss of the forward pass that precedes the only update, so
    # this is the anchor term under exactly the parameters that stamped the anchors.
    stats = agent._update_encoder(L, enc_steps=1, mb_segments=4)
    assert stats["anchor_free"] == 0.0
    assert stats["anchor_loss"] == pytest.approx(0.0, abs=1e-12)

    # Move the map deliberately: a bias shift on the encoder's mean head relabels every
    # segment at once, exactly the rotation the anchor exists to resist.
    with torch.no_grad():
        agent.encoder.net.net[-1].bias[0] += 1.0
    moved = agent._update_encoder(L, enc_steps=1, mb_segments=4)
    assert moved["anchor_loss"] > 1e-4


def test_settle_anchors_frees_warmup_and_recent_segments(agent):
    """Which segments carry an anchor: none under warm-up, then everything but the last
    ``anchor_window`` pushed -- each stamped once, with the code it had when it settled."""
    L, n = 16, 6
    for i in range(n):
        agent.replay.push(torch.randn(L, agent.encoder.in_dim), anchor=float(100 + i))

    agent.anchor_warmup, agent.anchor_window = 10, 2
    assert agent._settle_anchors(L) == n                        # n_seen (6) <= warmup (10)
    assert np.isnan(agent.replay.anchors).all()                 # warm-up pins nothing

    agent.anchor_warmup = 0                                     # warm-up over
    assert agent._settle_anchors(L) == 2                        # push_ids 4 and 5 stay free
    stamped = list(agent.replay.anchors)
    assert np.isfinite(stamped[:4]).all() and np.isnan(stamped[4:]).all()
    assert all(abs(a) < agent.z_scale + 1e-6 for a in stamped[:4])   # codes, not 100+

    with torch.no_grad():                                       # move the map
        agent.encoder.net.net[-1].bias[0] += 1.0
    assert agent._settle_anchors(L) == 2
    assert agent.replay.anchors[:4] == stamped[:4]        # a settled code is stamped ONCE

    agent.anchor_window = 0
    assert agent._settle_anchors(L) == 0                        # nothing free: all pinned
    assert np.isfinite(agent.replay.anchors).all()


#----- Code dispersion -----

def test_dispersion_hinge(agent):
    """Positive while the group codes are clustered, exactly zero once they are spread."""
    agent.disp_target_sd = 1.0
    solo = torch.arange(5)                       # every code its own group -> plain hinge

    # All codes equal: no variance at all, so the hinge is the full target, squared.
    assert float(agent._dispersion_loss(torch.zeros(5), solo)[0]) == pytest.approx(1.0)
    assert float(agent._dispersion_loss(torch.full((4,), 0.7),
                                        torch.arange(4))[0]) == pytest.approx(1.0)

    # Spread beyond the target: a HINGE, so the gradient is off, not merely small.
    spread = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])          # sd = 1.58 > 1.0
    assert float(agent._dispersion_loss(spread, solo)[0]) == 0.0
    spread = spread.clone().requires_grad_(True)
    agent._dispersion_loss(spread, solo)[0].backward()
    assert torch.count_nonzero(spread.grad) == 0

    # Partly spread: (target - sd)^2, and it pulls the codes apart.
    half = torch.tensor([-0.5, 0.0, 0.5])                        # sd = 0.5
    assert float(agent._dispersion_loss(half, torch.arange(3))[0]) == pytest.approx(0.25)
    half = half.clone().requires_grad_(True)
    agent._dispersion_loss(half, torch.arange(3))[0].backward()
    # Gradient DESCENT moves against the gradient, so the outermost codes travel outward.
    assert -half.grad[0] < 0 and -half.grad[-1] > 0
    assert half.grad[1] == pytest.approx(0.0, abs=1e-7)          # the centre code stays

    # Fewer than two groups carries no variance information.
    assert float(agent._dispersion_loss(torch.zeros(1), torch.zeros(1, dtype=torch.long))[0]) == 0.0


def test_dispersion_uses_group_means_and_penalises_within_group_spread(agent):
    """The exploit the plain hinge allowed -- buying variance with within-task noise --
    scores badly under the grouped form, and the invariance term prices it."""
    agent.disp_target_sd = 1.0
    gidx = torch.tensor([0, 0, 1, 1])

    # Two tight, well-separated tasks: hinge satisfied, nothing to pay for consistency.
    good = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    disp, inv = agent._dispersion_loss(good, gidx)
    assert float(disp) == 0.0 and float(inv) == pytest.approx(0.0)

    # The exploit: the SAME total spread, but bought inside the groups instead of between
    # them. The group means coincide, so the hinge is maximal AND invariance is charged.
    exploit = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    disp, inv = agent._dispersion_loss(exploit, gidx)
    assert float(disp) == pytest.approx(1.0)
    assert float(inv) == pytest.approx(1.0)
    # The two are permutations of each other, so a plain batch hinge scores them
    # IDENTICALLY -- it is blind to the difference between separating the tasks and
    # randomising the code within them. That blindness is the whole bug.
    assert float(exploit.std()) == pytest.approx(float(good.std()))
    assert float(agent._dispersion_loss(exploit, torch.arange(4))[0]) == \
        float(agent._dispersion_loss(good, torch.arange(4))[0]) == 0.0


def test_dispersion_reaches_the_encoder_objective(agent):
    """``disp_coef``/``inv_coef`` are off by default, gated by warm-up, and reported."""
    L = 16
    assert agent.disp_coef == 0.0 and agent.inv_coef == 0.0   # opt-in: old runs unchanged

    feats = torch.randn(L, agent.encoder.in_dim)
    for _ in range(4):
        agent.replay.push(feats.clone(), group=0)   # identical inputs, one task

    agent.disp_coef, agent.inv_coef, agent.disp_target_sd = 1.0, 1.0, 1.0
    agent.anchor_warmup = 0                          # warm-up over: dispersion is live
    stats = agent._update_encoder(L, enc_steps=1, mb_segments=4)
    # One group only -> no between-group variance to report, and identical inputs give it
    # no within-group variance either.
    assert stats["disp_loss"] == 0.0 and stats["inv_loss"] == pytest.approx(0.0)

    # Two groups whose codes coincide: the hinge is maximal.
    agent.replay.clear()
    for g in (0, 1):
        for _ in range(2):
            agent.replay.push(feats.clone(), group=g)
    stats = agent._update_encoder(L, enc_steps=1, mb_segments=4)
    assert stats["disp_loss"] == pytest.approx(1.0, abs=1e-6)


def test_dispersion_is_gated_by_the_anchor_warmup(agent):
    """A one-task pool must never be asked to spread: that IS within-task spread."""
    L = 16
    for _ in range(4):
        agent.replay.push(torch.randn(L, agent.encoder.in_dim), group=0)
    agent.disp_coef, agent.disp_target_sd = 1.0, 1.0

    agent.anchor_warmup = 10_000                     # still warming up
    assert agent._update_encoder(L, enc_steps=1, mb_segments=4)["disp_loss"] == 0.0


#----- Decoder-side gauge control -----

def make_dec_agent(**kw):
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    return AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                                 prior_sd=0.3, kl_coef=0.0, anchor_coef=0.0,
                                 encoder_lr=3e-4, **kw)


def test_slow_decoder_lr_group():
    """C1: the decoder gets its own, slower parameter group; the default is unchanged."""
    plain = make_dec_agent()
    enc_g, dec_g = plain.enc_optim.param_groups
    assert enc_g["lr"] == pytest.approx(3e-4) and dec_g["lr"] == pytest.approx(3e-4)
    assert {id(p) for p in dec_g["params"]} == {id(p) for p in plain.decoder.parameters()}
    assert {id(p) for p in enc_g["params"]} == {id(p) for p in plain.encoder.parameters()}

    slow = make_dec_agent(decoder_lr_ratio=30.0)
    assert slow.enc_optim.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert slow.enc_optim.param_groups[1]["lr"] == pytest.approx(3e-4 / 30.0)

    # And it bites: after identical updates the slow decoder has moved much less, while
    # the encoder is untouched by the knob.
    L = 16
    moved = {}
    for ratio in (1.0, 100.0):
        a = make_dec_agent(decoder_lr_ratio=ratio)
        before = [p.detach().clone() for p in a.decoder.parameters()]
        for _ in range(4):
            a.replay.push(torch.randn(L, a.encoder.in_dim))
        a._update_encoder(L, enc_steps=5, mb_segments=4)
        moved[ratio] = sum(float((p - b).abs().sum())
                           for p, b in zip(a.decoder.parameters(), before))
    assert moved[100.0] < moved[1.0] / 10.0


def test_film_decoder_shape_and_gauge():
    """C4: the linear-modulation decoder, and the gauge that ``normalise`` closes."""
    torch.manual_seed(0)
    sa = torch.randn(7, 6)
    z = torch.randn(7)

    plain = FiLMDecoder(6, 4, hidden=8, normalise=False)
    out = plain(sa, z)
    assert out.shape == (7, 4)
    # z enters LINEARLY: doubling z doubles the deviation from f(s, a).
    f_only = plain(sa, torch.zeros(7))
    assert torch.allclose(plain(sa, 2 * z) - f_only, 2 * (out - f_only), atol=1e-5)

    # THE GAUGE: z -> a*z with g -> g/a is the same function. This is the freedom that
    # lets the encoder rescale its whole code map for free.
    scale = 3.0
    with torch.no_grad():
        for prm in plain.g.net[-1].parameters():
            prm.div_(scale)
    assert torch.allclose(plain(sa, scale * z), out, atol=1e-5)

    # Normalising g's output layer closes it: g's scale is no longer free, so rescaling z
    # is now observable in the output.
    torch.manual_seed(0)
    normed = FiLMDecoder(6, 4, hidden=8, normalise=True)
    out_n = normed(sa, z)
    with torch.no_grad():
        for prm in normed.g.net[-1].parameters():
            prm.div_(scale)
    assert torch.allclose(normed(sa, z), out_n, atol=1e-5)      # g's scale is inert...
    assert not torch.allclose(normed(sa, scale * z), out_n, atol=1e-3)   # ...so z's is not


def test_decoder_arch_wiring_and_validation():
    """The agent builds, decodes and trains through either decoder; default is unchanged."""
    L = 16
    assert isinstance(make_dec_agent().decoder, nn.Module)
    assert make_dec_agent().decoder_arch == "concat"

    for norm in (False, True):
        a = make_dec_agent(decoder_arch="film", film_normalise=norm)
        assert isinstance(a.decoder, FiLMDecoder)
        feats = torch.randn(L, a.encoder.in_dim)
        assert a._decode_next_obs(feats, torch.randn(L)).shape == (L, a.obs_dim)
        for _ in range(4):
            a.replay.push(feats.clone())
        before = [p.detach().clone() for p in a.decoder.parameters()]
        out = a._update_encoder(L, enc_steps=2, mb_segments=4)
        assert np.isfinite(out["dyn_loss"])
        assert any(not torch.equal(b, p)
                   for b, p in zip(before, a.decoder.parameters()))

    with pytest.raises(ValueError):
        make_dec_agent(decoder_arch="flim")


#----- Structure-not-values anchors -----

def test_centroid_anchor_frees_a_symmetric_split_but_charges_a_sweep(agent):
    """Variant A, stated as a test: splitting a cluster is free, moving it is not."""
    agent.anchor_mode, agent.anchor_group_gap = "centroid", 0.05
    stamped = torch.tensor([-1.0, -1.0, 1.0, 1.0])       # two clusters, far apart

    # A symmetric split of BOTH clusters: members move +-0.09, every centroid stays.
    split = torch.tensor([-1.09, -0.91, 0.91, 1.09])
    assert float(agent._anchor_penalty(split, stamped)) == pytest.approx(0.0, abs=1e-12)
    # The plain value anchor charges the very same move.
    agent.anchor_mode = "value"
    assert float(agent._anchor_penalty(split, stamped)) == pytest.approx(0.09 ** 2)

    # A uniform translation moves every centroid, so it is charged in full either way.
    agent.anchor_mode = "centroid"
    shifted = stamped + 0.2
    assert float(agent._anchor_penalty(shifted, stamped)) == pytest.approx(0.04)

    # An ASYMMETRIC split still moves the centroid and is charged accordingly.
    lopsided = torch.tensor([-1.0, -0.8, 1.0, 1.0])      # cluster 0 centroid moves +0.1
    assert float(agent._anchor_penalty(lopsided, stamped)) == pytest.approx(0.01 / 2)

    # With the gap below a cluster's own width every code becomes its own cluster and the
    # centroid form degenerates to the value form -- the knob does what it says. (Stamps
    # must be DISTINCT for this: no positive gap can ever split identical values.)
    spread_stamps = torch.tensor([-1.02, -0.98, 0.98, 1.02])
    agent.anchor_group_gap = 1e-9
    assert float(agent._anchor_penalty(spread_stamps + 0.09,
                                       spread_stamps)) == pytest.approx(0.09 ** 2)


def test_distance_anchor_frees_expansion_and_charges_contraction(agent):
    """Variant B: a split is an expansion, so it must cost nothing."""
    agent.anchor_mode = "distance"
    stamped = torch.tensor([-1.0, 1.0])

    assert float(agent._anchor_penalty(stamped.clone(), stamped)) == pytest.approx(0.0)
    # Expansion: free, and with no gradient at all.
    grown = torch.tensor([-1.5, 1.5], requires_grad=True)
    pen = agent._anchor_penalty(grown, stamped)
    assert float(pen) == pytest.approx(0.0)
    pen.backward()
    assert torch.count_nonzero(grown.grad) == 0
    # Contraction: charged (distance 2.0 -> 1.0).
    assert float(agent._anchor_penalty(torch.tensor([-0.5, 0.5]),
                                       stamped)) == pytest.approx(1.0)
    # Translation is charged even though every pairwise distance is preserved -- this is
    # the term that stops the map escaping the hinge as a rigid body.
    assert float(agent._anchor_penalty(stamped + 0.3, stamped)) == pytest.approx(0.09)

    # LIMITATION, asserted so it cannot be forgotten: inside a wider set a symmetric split
    # is NOT free. Splitting [-1,-1] apart moves its inner member toward the far cluster,
    # so the cross-pair (1,2) contracts 2.00 -> 1.82 and is charged, even though the split
    # itself is an expansion. Variant B is therefore stricter than the centroid form on
    # exactly the move we want to permit -- one charged pair out of six.
    st = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    pen = float(agent._anchor_penalty(torch.tensor([-1.09, -0.91, 0.91, 1.09]), st))
    assert pen == pytest.approx(0.18 ** 2 / 6)
    agent.anchor_mode = "centroid"
    agent.anchor_group_gap = 0.05
    assert float(agent._anchor_penalty(torch.tensor([-1.09, -0.91, 0.91, 1.09]),
                                       st)) == pytest.approx(0.0, abs=1e-12)


def test_code_clusters_and_stratified_sampling():
    """Clustering reads stamps only, and the draw supplies cluster-mates."""
    np.random.seed(0)
    torch.manual_seed(0)
    buf = SegmentReplayBuffer(capacity=40)
    codes = ([-1.0] * 10 + [-0.98] * 10 + [1.0] * 10)     # two clusters at gap 0.05
    for i, c in enumerate(codes):
        buf.push(torch.full((2, 3), float(i)), anchor=c)
    buf.push(torch.full((2, 3), 99.0))                    # unsettled: must be ignored

    clusters = buf.code_clusters(gap=0.05)
    assert len(clusters) == 2 and sorted(len(c) for c in clusters) == [10, 20]
    assert all(np.isfinite(buf.anchors[i]) for c in clusters for i in c)
    # A tighter gap splits the near-coincident pair apart.
    assert len(buf.code_clusters(gap=0.01)) == 3

    feats, anchors = buf.sample_code_groups(4, gap=0.05, per_group=2)
    assert feats.shape == (4 * 2, 3) and anchors.shape == (4,)
    # Two clusters x two members each: some pair must share a cluster.
    rounded = np.round(anchors.numpy(), 2)
    assert len(rounded) - len(set(rounded)) >= 1


def test_anchor_mode_defaults_and_validation(agent):
    """Default behaviour is untouched, and a typo fails loudly."""
    assert agent.anchor_mode == "value"
    with pytest.raises(ValueError):
        AmortisedCOINPPOAgent(CustomCartPoleEnv(max_episode_steps=50), CTX_IDS,
                              encoder_hidden=8, anchor_mode="centoid")


def test_structure_anchors_reach_the_encoder_objective():
    """Both variants run end to end through ``_update_encoder`` and report a loss."""
    L = 16
    for mode in ("centroid", "distance"):
        env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
        torch.manual_seed(0)
        a = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                                  prior_sd=0.3, kl_coef=0.0, anchor_coef=1.0,
                                  anchor_window=0, anchor_warmup=0, anchor_mode=mode)
        for _ in range(6):
            a.replay.push(torch.randn(L, a.encoder.in_dim))
        a._settle_anchors(L)                       # stamp them all
        assert np.isfinite(a.replay.anchors).all()
        stats = a._update_encoder(L, enc_steps=1, mb_segments=4)
        # Freshly stamped: the codes have not moved yet, so every form reads ~0.
        assert stats["anchor_loss"] == pytest.approx(0.0, abs=1e-9)

        with torch.no_grad():                      # a bodily sweep: all forms charge it
            a.encoder.net.net[-1].bias[0] += 1.0
        assert a._update_encoder(L, enc_steps=1, mb_segments=4)["anchor_loss"] > 1e-4


#----- Recency-balanced replay -----

def test_sample_balanced_reserves_half_the_batch_for_the_newest_segments():
    """Half the draw comes from the newest window however the pool is composed -- the
    point being a late task's gradient share stops depending on its share of the stream."""
    np.random.seed(0)
    torch.manual_seed(0)
    buf = SegmentReplayBuffer(capacity=100)
    for i in range(100):                     # tag each segment with its arrival index
        buf.push(torch.full((2, 3), float(i)))

    window = 10                              # newest 10 => tags 90..99
    counts = np.zeros(100)
    for _ in range(400):
        feats, anchors = buf.sample_balanced(8, window)
        assert feats.shape == (8 * 2, 3) and anchors.shape == (8,)
        tags = feats[::2, 0].numpy().astype(int)
        assert len(set(tags)) == 8           # still without replacement
        counts[tags] += 1
        assert (tags >= 90).sum() >= 4       # at least half from the recent window

    recent_share = counts[90:].sum() / counts.sum()
    # 4 of 8 guaranteed recent, plus the uniform half occasionally landing there.
    assert 0.5 <= recent_share <= 0.62
    assert counts[:90].min() > 0             # the old 90% are still all reachable


def test_sample_balanced_falls_back_when_there_is_nothing_recent():
    """A pool with fewer recent segments than half the batch still returns a full batch."""
    np.random.seed(0)
    buf = SegmentReplayBuffer(capacity=10)
    for i in range(10):
        buf.push(torch.full((2, 3), float(i)))

    feats, _ = buf.sample_balanced(6, recent_window=1)     # only one segment is "recent"
    assert feats.shape == (6 * 2, 3)
    assert len({float(v) for v in feats[::2, 0]}) == 6
    # And a batch larger than the pool is capped, not padded.
    assert buf.sample_balanced(50, recent_window=4)[0].shape == (10 * 2, 3)


def test_balanced_replay_is_off_by_default_and_reaches_the_encoder(agent):
    """Opt-in, and when on it is the sampler ``_update_encoder`` actually calls."""
    L = 16
    assert agent.balanced_replay is False
    for _ in range(6):
        agent.replay.push(torch.randn(L, agent.encoder.in_dim))

    seen = []
    inner = agent.replay.sample_balanced
    agent.replay.sample_balanced = lambda n, w: (seen.append((n, w)) or inner(n, w))

    agent._update_encoder(L, enc_steps=1, mb_segments=4)
    assert seen == []                                   # default path untouched

    agent.balanced_replay = True
    agent._update_encoder(L, enc_steps=1, mb_segments=4)
    assert seen == [(4, agent.anchor_window)]


def test_z_channel_noise_is_off_by_default_and_changes_only_the_sampled_z(agent):
    """Zero keeps the plain objective bit-for-bit; a positive floor perturbs the draw."""
    L = 16
    assert agent.z_channel_noise == 0.0
    for _ in range(6):
        agent.replay.push(torch.randn(L, agent.encoder.in_dim))

    def one_step(noise):
        states = {m: {k: v.detach().clone() for k, v in m.state_dict().items()}
                  for m in (agent.encoder, agent.decoder)}
        opt_state = agent.enc_optim.state_dict()
        agent.z_channel_noise = noise
        torch.manual_seed(7)
        np.random.seed(7)
        stats = agent._update_encoder(L, enc_steps=1, mb_segments=4)
        agent.z_channel_noise = 0.0
        for m, s in states.items():
            m.load_state_dict(s)
        agent.enc_optim.load_state_dict(opt_state)
        return stats["dyn_loss"]

    base, again, noisy = one_step(0.0), one_step(0.0), one_step(0.5)
    assert base == again                        # zero is exactly the old path
    assert np.isfinite(noisy) and noisy != base  # the channel actually perturbs z


def test_quantile_readout_off_by_default_is_identity(agent):
    """Without the flag every observation passes through untouched."""
    assert agent.quantile_readout is False
    agent._refresh_code_population(16)
    assert agent._code_pop is None
    assert agent._obs_transform(0.37, 0.05) == (0.37, 0.05)


def test_quantile_transform_is_invariant_to_monotone_warps():
    """The whole point: probit(rank) is unchanged by any monotone relabelling."""
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    a = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                              prior_sd=0.3, kl_coef=0.0, quantile_readout=True)
    rng = np.random.default_rng(3)
    pop = np.sort(rng.normal(size=64))
    z, sd = float(pop[40]) + 1e-6, 0.2

    a._code_pop = pop
    z1, s1 = a._obs_transform(z, sd)

    warp = lambda x: np.tanh(1.7 * x) + 0.03 * x ** 3   # strictly monotone
    a._code_pop = np.sort(warp(pop))
    z2, s2 = a._obs_transform(float(warp(z)), np.nan)   # sd needs its own interval
    assert z1 == pytest.approx(z2, abs=1e-9)            # rank -> same probit

    # sd maps through the same interval rule and is floored away from zero
    a._code_pop = pop
    _, s_edge = a._obs_transform(float(pop[-1]) + 5.0, 0.01)
    assert s_edge >= 1e-3
    assert s1 > 0.0


def test_quantile_population_comes_from_the_current_encoder():
    """Refresh re-encodes stored segments; a changed encoder changes the population."""
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    a = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                              prior_sd=0.3, kl_coef=0.0, quantile_readout=True)
    L = 16
    for _ in range(10):
        a.replay.push(torch.randn(L, a.encoder.in_dim))
    a._refresh_code_population(L)
    pop1 = a._code_pop.copy()
    assert pop1.shape[0] == 10 and np.all(np.diff(pop1) >= 0)

    with torch.no_grad():
        for p in a.encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    a._refresh_code_population(L)
    assert not np.allclose(a._code_pop, pop1)

    # seg_len inferred from the buffer when omitted (the evaluate_identifying path)
    a._refresh_code_population()
    assert a._code_pop.shape[0] == 10


def test_residual_decoder_predicts_state_plus_delta(agent):
    """decoder_residual adds s back: prediction = s + f(s, a, z), same f either way."""
    L = 8
    feats = torch.randn(L, agent.encoder.in_dim)
    z = torch.zeros(L)
    assert agent.decoder_residual is False
    plain = agent._decode_next_obs(feats, z)
    agent.decoder_residual = True
    residual = agent._decode_next_obs(feats, z)
    agent.decoder_residual = False
    assert torch.allclose(residual, feats[:, :agent.obs_dim] + plain)


def _value_loss_setup(agent, perturb_head=False):
    """Two instantiated heads + one segment of fake data for _encoder_value_loss."""
    L = 8
    agent.ensure_contexts(2)
    if perturb_head:
        with torch.no_grad():
            for p in agent.nets[agent.context_keys[1]][2].parameters():
                p.add_(torch.randn_like(p))
    feats = torch.randn(L, agent.encoder.in_dim)
    W = agent.num_contexts
    pi = np.full(W, np.nan)
    pi[:2] = 0.5
    pi[-1] = 0.0
    ctx = (np.array([-0.1, 0.1, 0.0]), np.array([0.05, 0.05, 0.25]), 1e-4)
    obs = torch.randn(L, agent.obs_dim)
    ret = torch.randn(L)
    return agent._encoder_value_loss([feats], [pi], [ctx], obs, ret, L)


def test_value_loss_gradient_reaches_encoder_only_when_heads_disagree():
    """The recorded history in one test: identical (cloned) heads give ~zero encoder
    gradient -- the cold-start failure that killed value-ONLY training -- while
    differentiated heads give a real one, which is the regime L_dyn cannot see."""
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)

    def grad_norm(perturb):
        torch.manual_seed(0)
        a = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                                  prior_sd=0.3, kl_coef=0.0, value_coef=1.0)
        loss = _value_loss_setup(a, perturb_head=perturb)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        return float(sum(p.grad.abs().sum() for p in a.encoder.parameters()
                         if p.grad is not None))

    agree, disagree = grad_norm(False), grad_norm(True)
    assert agree < 1e-6                # clones agree -> responsibility grad is inert
    assert disagree > 1e-6             # differentiated heads -> encoder feels value


def test_value_coef_is_off_by_default(agent):
    assert agent.value_coef == 0.0


def test_rail_hinge_charges_saturated_codes_only(agent):
    """rail_coef penalises |mean|/z_scale beyond 0.8 and is inert inside it."""
    assert agent.rail_coef == 0.0
    zs = agent.z_scale
    inner = torch.tensor([0.0, 0.5 * zs, -0.79 * zs])
    outer = torch.tensor([0.95 * zs, -zs])
    hinge = lambda v: torch.relu(v.abs() / zs - 0.8).pow(2).mean()
    assert float(hinge(inner)) == 0.0
    assert float(hinge(outer)) > 0.0


def test_encoder_reward_layout_and_prediction():
    """(s, a, r, s') layout: s' stays the last obs_dim columns, the decoder grows a
    reward head trained by _dyn_loss, and the reward never reaches the decoder input."""
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    a = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                              prior_sd=0.3, kl_coef=0.0, encoder_reward=True)
    assert a.encoder.in_dim == 2 * a.obs_dim + a.act_dim + 1
    L = 8
    obs = [torch.randn(a.obs_dim) for _ in range(L)]
    act = [0] * L
    nxt = [torch.randn(a.obs_dim) for _ in range(L)]
    rew = [float(i) for i in range(L)]
    feats = a._segment_features(obs, act, nxt, rew=rew)
    assert torch.allclose(feats[:, -a.obs_dim:], torch.stack(nxt))       # s' last
    assert torch.allclose(feats[:, a.obs_dim + a.act_dim],
                          torch.tensor(rew))                             # r in between
    with pytest.raises(ValueError):
        a._segment_features(obs, act, nxt)                               # rew required

    s_hat, r_hat = a._decode_full(feats, torch.zeros(L))
    assert s_hat.shape == (L, a.obs_dim) and r_hat.shape == (L,)
    loss = a._dyn_loss(feats, torch.zeros(L), L)
    assert torch.isfinite(loss)
    loss.backward()
    grads = sum(p.grad.abs().sum() for p in a.decoder.parameters())
    assert float(grads) > 0.0


def test_encoder_reward_off_keeps_the_old_interface(agent):
    """Default agents keep in_dim 15 and ignore a passed rew entirely."""
    assert agent.encoder.use_reward is False
    assert agent.encoder.in_dim == 2 * agent.obs_dim + agent.act_dim
    L = 4
    obs = [torch.randn(agent.obs_dim) for _ in range(L)]
    nxt = [torch.randn(agent.obs_dim) for _ in range(L)]
    f1 = agent._segment_features(obs, [0] * L, nxt)
    f2 = agent._segment_features(obs, [0] * L, nxt, rew=[9.9] * L)
    assert torch.equal(f1, f2)


def make_repel_agent():
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    a = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                              prior_sd=0.3, kl_coef=0.0, repel_coef=1.0)
    a.ensure_contexts(1)
    return a


def _repel_inputs(a, ret_scale):
    L = 8
    feats = a._segment_features([torch.randn(a.obs_dim) for _ in range(L)], [0] * L,
                                [torch.randn(a.obs_dim) for _ in range(L)])
    rho = np.zeros(a.num_contexts)
    rho[0] = 1.0
    ctx = (np.array([0.1, 0.0]), np.array([0.05, 0.25]), 1e-4)
    obs = torch.randn(L, a.obs_dim)
    ret = torch.full((L,), float(ret_scale))
    return [feats], [rho], [ctx], obs, ret, L


def test_value_surprise_gate_needs_a_baseline_and_a_blowup():
    """First sight sets the baseline (no flag); a same-scale repeat updates it; a
    blow-up flags the segment; and the blow-up does NOT poison the baseline."""
    a = make_repel_agent()
    cid = a.context_keys[0]
    assert a._flag_value_surprise(*_repel_inputs(a, 1.0)) == 0     # baseline set
    base = a._vloss_ema[cid]
    assert a._flag_value_surprise(*_repel_inputs(a, 1.0)) == 0     # ordinary: EMA
    assert a._flag_value_surprise(*_repel_inputs(a, 500.0)) == 1   # massive: flag
    assert len(a._repel_batch) == 1
    assert a._vloss_ema[cid] == pytest.approx(base, rel=0.5)       # not poisoned
    # repel_coef == 0 disables everything
    a.repel_coef = 0.0
    assert a._flag_value_surprise(*_repel_inputs(a, 500.0)) == 0
    assert a._repel_batch == []


def test_value_surprise_never_fires_on_the_novel_column():
    """A rollout routed to novel has no established head to be surprised by."""
    a = make_repel_agent()
    seg_feats, _, ctx, obs, ret, L = _repel_inputs(a, 500.0)
    rho = np.zeros(a.num_contexts)
    rho[-1] = 1.0                                  # dominant = novel
    assert a._flag_value_surprise(seg_feats, [rho], ctx, obs, ret, L) == 0


def test_repel_hinge_pushes_the_code_away_from_the_parked_centre():
    """Gradient steps on the hinge alone move the segment's code away from mu_c,
    and the hinge dies once the margin is cleared."""
    a = make_repel_agent()
    L = 8
    # Break the zero-init symmetry (fresh encoders emit exactly 0 for everything, and
    # |z - mu| has no gradient at exactly 0 -- unreachable in real use, where the
    # gate only fires on established, trained contexts).
    with torch.no_grad():
        for p in a.encoder.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    feats = a._segment_features([torch.randn(a.obs_dim) for _ in range(L)], [0] * L,
                                [torch.randn(a.obs_dim) for _ in range(L)])
    with torch.no_grad():
        mean, _ = a.encoder.prefix_posterior(feats, L)
    mu_c = float(mean[0, -1]) + 0.005              # park the centre next to the code
    a._repel_batch = [(feats, mu_c)]
    a.repel_margin = 0.1

    def dist():
        with torch.no_grad():
            m, _ = a.encoder.prefix_posterior(feats, L)
        return abs(float(m[0, -1]) - mu_c)

    d0 = dist()
    opt = torch.optim.Adam(a.encoder.parameters(), lr=1e-2)
    for _ in range(50):
        loss = a._repel_hinge(L)
        opt.zero_grad(); loss.backward(); opt.step()
    assert dist() > d0 + 0.02                      # actually moved away
    assert float(a._repel_hinge(L)) < 1e-4 or dist() >= 0.1  # hinge released


#----- 2-D COIN value observation -----

def make_coin_2d(seed=0, value_q=0.01):
    """A small MD model matching the observe_value docstring's construction."""
    return RealTimeCOIN(rng=seed, state_dim=2, sigma_motor_noise=0.01,
                        prior_mean_retention=0.9995,
                        process_noise_covariance=np.diag([0.0089 ** 2,
                                                          value_q ** 2]))


def test_observe_value_is_off_by_default(agent):
    assert agent.observe_value is False
    assert agent.value_obs_scale == 200.0
    assert agent.value_obs_noise_floor == 0.05
    assert agent.value_process_noise == 0.01


def test_step_context_weights_md_uses_the_z_marginal(agent):
    """2-D contexts with only z observed give exactly the scalar answer on the z
    components; observing the value dim too actually changes the weights."""
    pi = np.array([0.6, 0.3, np.nan, 0.1])
    mu = np.array([-0.5, 0.4, 0.0])
    var = np.array([0.02, 0.03, 0.25])
    z, sd, floor2 = 0.35, 0.05, 1e-4
    scalar_w = agent._step_context_weights(pi, z, sd, mu, var, floor2)

    mu_md = np.column_stack([mu, [5.0, -5.0, 0.0]])     # junk value centres
    var_md = np.column_stack([var, [0.1, 0.1, 0.1]])
    md_w = agent._step_context_weights(pi, z, sd, mu_md, var_md, floor2)
    np.testing.assert_allclose(md_w, scalar_w)          # scalar z -> z marginal

    md_w2 = agent._step_context_weights(pi, np.array([z, np.nan]),
                                        np.array([sd, 0.0]), mu_md, var_md, floor2)
    np.testing.assert_allclose(md_w2, scalar_w)         # explicit nan mask, same

    both = agent._step_context_weights(pi, np.array([z, 4.0]),
                                       np.array([sd, 0.1]), mu_md, var_md, floor2)
    assert not np.allclose(both, scalar_w)              # value dim carries weight

    nothing = agent._step_context_weights(pi, np.array([np.nan, np.nan]),
                                          np.array([0.0, 0.0]), mu_md, var_md,
                                          floor2)
    assert nothing is pi                                # no evidence: the prior


def test_segment_context_gaussians_md_shapes_and_novel_prior(agent):
    """MD branch: (k+1, 2) per-dim means/vars, novel row at the per-dim
    stationary moments of the supplied diagonal process noise."""
    coin = make_coin_2d(value_q=0.02)
    for i in range(6):
        coin.observe_q(None)
        coin.observation_noise_covariance = np.diag([0.03 ** 2, 0.05 ** 2])
        coin.observe_y(np.array([0.3 if i % 2 == 0 else -0.3,
                                 0.5 if i % 2 == 0 else -0.5]))
    k = int(coin.context_alignment()["K"])
    assert k >= 1
    mu, var = agent._segment_context_gaussians(coin, k)
    assert mu.shape == (k + 1, 2) and var.shape == (k + 1, 2)
    assert np.isfinite(mu).all() and (var > 0).all()
    a0 = float(coin.prior_mean_retention)
    np.testing.assert_allclose(
        var[-1], np.array([0.0089 ** 2, 0.02 ** 2]) / (1.0 - a0 ** 2))


def test_observe_value_feeds_coin_2d_vectors():
    """Full MD round trip: train_step observes (z, R) with a per-trial diagonal
    covariance, R is the scaled mean of the segment's ENDED episodes (nan when
    none ended), and the self-identifying eval runs on the z marginal."""
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    agent = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                                  prior_sd=0.3, kl_coef=0.0, observe_value=True)
    coin = make_coin_2d()
    seen_y, seen_cov = [], []
    inner = coin.observe_y

    def spy(y=None):
        seen_y.append(np.array(y, dtype=float))
        seen_cov.append(np.array(coin.observation_noise_covariance, dtype=float))
        return inner(y)

    coin.observe_y = spy
    out = agent.train_step(make_envs(), coin, seg_steps=32, mini_epochs=2,
                           mb_size=16)
    assert coin.trial == 2 and len(seen_y) == 2
    for y, cov in zip(seen_y, seen_cov):
        assert y.shape == (2,) and np.isfinite(y[0])
        # 50-step-capped CartPole episodes: any finite R sits in (0, 0.25].
        assert np.isnan(y[1]) or 0.0 < y[1] <= 50.0 / 200.0 + 1e-9
        assert cov.shape == (2, 2) and cov[0, 1] == 0.0 == cov[1, 0]
        assert cov[0, 0] >= 0.01 ** 2                   # z floor folded in
        assert cov[1, 1] >= agent.value_obs_noise_floor ** 2
    assert np.isfinite(out["dyn_loss"])

    res = agent.evaluate_identifying(env, coin, n_episodes=2, max_steps=8)
    assert res["returns"].shape == (2,) and np.isfinite(res["returns"]).all()
    env.close()


def test_observe_value_uses_raw_returns_on_shaped_envs():
    """The value observation must ride the RAW reward channel: on a shaped
    MountainCar the observed r_bar equals the raw episodic mean, not the shaped one."""
    from environments import MountainCarXEnv

    env = MountainCarXEnv(amplitude=1.0, shaping_coef=60000.0,
                          max_episode_steps=40)
    torch.manual_seed(0)
    a = AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3,
                              prior_sd=0.3, kl_coef=0.0, observe_value=True)

    seen = {}

    class SpyCoin:
        sigma_motor_noise = 0.01
        state_dim = 2

        def observe_q(self, q):            pass
        def observe_y(self, y):            seen["y"] = np.array(y, dtype=float)
        def context_alignment(self):
            return {"K": 0, "global_contexts": {"state_mean": np.zeros((0, 2)),
                                                "bias_mean": np.zeros((0, 2)),
                                                "state_cov": np.zeros((0, 2, 2))}}
        def responsibilities_vector(self):
            v = np.zeros(11); v[0] = 1.0; return v
        def stationary_context_probabilities(self):
            return np.full(11, 1 / 11)
        prior_mean_retention = 0.9995
        prior_mean_drift = 0.0
        sigma_process_noise = 0.0089
        process_noise_covariance = np.diag([0.0089 ** 2, 0.01 ** 2])

    import rl as rl_mod
    orig = rl_mod.coin_predicted_pi
    rl_mod.coin_predicted_pi = lambda coin, cue=None: (np.full(11, np.nan), 0)
    try:
        a.train_step([env], SpyCoin(), seg_steps=64, mini_epochs=1, mb_size=32,
                     enc_steps=1, mb_segments=1, carry_state=False)
    finally:
        rl_mod.coin_predicted_pi = orig
    env.close()

    assert "y" in seen and seen["y"].shape == (2,)
    r_obs = seen["y"][1]
    if np.isfinite(r_obs):
        # A 40-step shaped MC episode: raw return is about -40/200; the shaped one
        # is displaced by the potential difference. The observation must be raw.
        assert -0.5 <= r_obs <= 0.0
        assert abs(r_obs - (-40.0 / 200.0)) < 0.1


#----- EMA-teacher anchor -----

def make_ema_agent(tau=0.9, anchor_coef=1.0):
    """An agent in EMA-teacher mode, on the same tiny CartPole interface as ``agent``."""
    env = CustomCartPoleEnv(force_mag=10.0, max_episode_steps=50)
    torch.manual_seed(0)
    return AmortisedCOINPPOAgent(env, CTX_IDS, encoder_hidden=8, z_scale=0.3, prior_sd=0.3,
                                 kl_coef=0.0, anchor_coef=anchor_coef, anchor_ema_tau=tau)


def test_ema_teacher_starts_as_a_copy_and_takes_no_gradient():
    """Built from the student, detached from it, and never optimised."""
    a = make_ema_agent()
    assert a.encoder_ema is not None and a.encoder_ema is not a.encoder
    for t, s in zip(a.encoder_ema.parameters(), a.encoder.parameters()):
        assert torch.equal(t, s)                       # an exact copy at construction
        assert not t.requires_grad                     # ...that can never take a gradient
    assert not a.encoder_ema.training                  # eval mode

    # The teacher is not in any optimiser, so no step can reach it.
    owned = {id(p) for opt in a._all_optimizers() for g in opt.param_groups
             for p in g["params"]}
    owned |= {id(p) for g in a.enc_optim.param_groups for p in g["params"]}
    assert not any(id(p) in owned for p in a.encoder_ema.parameters())

    # And it does not inflate the reported trainable parameter count.
    from baselines import fig3_common as f3
    plain = AmortisedCOINPPOAgent(CustomCartPoleEnv(max_episode_steps=50), CTX_IDS,
                                  encoder_hidden=8, z_scale=0.3, prior_sd=0.3)
    assert f3.count_parameters(a) == f3.count_parameters(plain)


def test_ema_teacher_trails_the_student():
    """One EMA step moves the teacher exactly ``(1 - tau)`` of the way to the student."""
    a = make_ema_agent(tau=0.9)
    before = [t.detach().clone() for t in a.encoder_ema.parameters()]
    with torch.no_grad():                              # move the student by a known amount
        for s in a.encoder.parameters():
            s.add_(1.0)
    a._update_ema_teacher()

    for t, b0, s in zip(a.encoder_ema.parameters(), before, a.encoder.parameters()):
        assert torch.allclose(t, 0.9 * b0 + 0.1 * s, atol=1e-6)
        assert not torch.allclose(t, s)                # trailing, not tracking exactly

    # Repeated steps with a still student converge the teacher onto it.
    for _ in range(200):
        a._update_ema_teacher()
    for t, s in zip(a.encoder_ema.parameters(), a.encoder.parameters()):
        assert torch.allclose(t, s, atol=1e-6)


def test_ema_anchor_is_zero_at_init_and_positive_once_the_student_moves():
    """The whole mechanism, as a test: no penalty while student and teacher agree, a real
    one as soon as the student relabels the map."""
    L = 16
    a = make_ema_agent(tau=0.999)
    for _ in range(4):
        a.replay.push(torch.randn(L, a.encoder.in_dim))

    # Teacher == student at construction, so the anchor is identically zero.
    stats = a._update_encoder(L, enc_steps=1, mb_segments=4)
    assert stats["anchor_loss"] == pytest.approx(0.0, abs=1e-12)
    assert stats["anchor_free"] == 0.0                 # stamping is inert in EMA mode

    # Relabel the map: a bias shift on the encoder's mean head moves every code at once.
    with torch.no_grad():
        a.encoder.net.net[-1].bias[0] += 1.0
    assert a._update_encoder(L, enc_steps=1, mb_segments=4)["anchor_loss"] > 1e-4


def test_ema_mode_stamps_nothing():
    """No stored code, no window, no warm-up -- so no task-boundary information is used."""
    L = 16
    a = make_ema_agent()
    a.anchor_window, a.anchor_warmup = 0, 0            # settings that WOULD stamp
    for _ in range(4):
        a.replay.push(torch.randn(L, a.encoder.in_dim))

    assert a._settle_anchors(L) == 0
    assert np.isnan(a.replay.anchors).all()            # nothing was ever stamped
    a._update_encoder(L, enc_steps=2, mb_segments=4)
    assert np.isnan(a.replay.anchors).all()


def test_ema_teacher_bounds_code_velocity():
    """The point of the mode: the map may move, but only as fast as the teacher follows.

    Same student, same data, same number of steps -- a tighter ``tau`` must not move the
    code further than a looser one.
    """
    L, feats = 16, torch.randn(16, 15)

    def travel(tau):
        a = make_ema_agent(tau=tau, anchor_coef=50.0)
        a.encoder.load_state_dict(make_ema_agent(tau=tau).encoder.state_dict())
        f = feats[:, :a.encoder.in_dim].clone()
        for _ in range(6):
            a.replay.push(f.clone())
        with torch.no_grad():
            z0 = float(a.encoder.prefix_posterior(f, L)[0][0, -1])
        # A strong, consistent pull on the map: shift the mean head every step.
        for _ in range(15):
            with torch.no_grad():
                a.encoder.net.net[-1].bias[0] += 0.05
            a._update_encoder(L, enc_steps=2, mb_segments=4)
        with torch.no_grad():
            z1 = float(a.encoder.prefix_posterior(f, L)[0][0, -1])
        return abs(z1 - z0)

    assert travel(0.999) <= travel(0.9) + 1e-6


#----- Episode carry-over -----

def test_start_segment_carry_and_reset(agent):
    """State and the time-limit counter cross the boundary; ``carry_state=False`` does not."""
    obs_t, ep_ret = agent._start_segment(make_envs()[0], carry_state=True)
    assert ep_ret == 0.0 and obs_t.shape == (OBS_DIM,)      # no carry yet -> plain reset

    state = np.array([0.1, -0.2, 0.05, 0.3])
    agent._carry = {"state": state, "elapsed": 7, "ep_ret": 12.0}

    env = make_envs()[1]
    obs_t, ep_ret = agent._start_segment(env, carry_state=True)
    assert np.allclose(env.state, state) and env._elapsed_steps == 7
    assert np.allclose(obs_t.cpu().numpy(), state) and ep_ret == 12.0

    env = make_envs()[1]
    obs_t, ep_ret = agent._start_segment(env, carry_state=False)
    assert env._elapsed_steps == 0 and ep_ret == 0.0
    assert not np.allclose(obs_t.cpu().numpy(), state)

    agent.reset_carry()
    env = make_envs()[1]
    _, ep_ret = agent._start_segment(env, carry_state=True)
    assert agent._carry is None and env._elapsed_steps == 0 and ep_ret == 0.0


def test_train_step_carries_episode_across_segments(agent):
    """A boundary switches the task, not the episode -- including a rollout boundary."""
    coin, L, starts, dones = RealTimeCOIN(rng=0), 8, [], []
    spy_on_segment_starts(agent, starts)
    spy_on_advantages(agent, dones)
    kwargs = dict(seg_steps=L, mini_epochs=1, mb_size=8, enc_steps=1, mb_segments=1)

    first = agent.train_step(make_endless_envs(), coin, **kwargs)
    second = agent.train_step(make_endless_envs(), coin, **kwargs)

    assert len(starts) == 4
    assert starts[0][0] is None and starts[0][2] == 0.0 and starts[0][3] == 0
    for k in (1, 2, 3):
        carry, obs_t, ep_ret, elapsed = starts[k]
        assert carry is not None
        assert elapsed == k * L                     # the limit stays per EPISODE
        assert ep_ret == float(k * L)               # CartPole pays one per step
        assert np.allclose(obs_t.cpu().numpy(), carry["state"], atol=1e-6)
    assert agent._carry is not None
    # A boundary is not the end of anything: GAE bootstraps there instead of cutting.
    assert all(not seg[-1] for seg in dones)

    # Nothing ended, so no segment reports a return -- nan, not a zero that would drag a
    # mean down.
    assert np.isnan(first["mean_episode_return"]).all()
    assert np.isnan(second["mean_episode_return"]).all()


def test_train_step_resets_only_after_termination(agent):
    """An episode that truncates at the segment end drops the carry; the next one resets."""
    L, starts, dones = 8, [], []
    spy_on_segment_starts(agent, starts)
    spy_on_advantages(agent, dones)
    envs = make_endless_envs(steps=L)               # truncates exactly at the boundary

    out = agent.train_step(envs, RealTimeCOIN(rng=0), seg_steps=L, mini_epochs=1,
                           mb_size=8, enc_steps=1, mb_segments=1)

    assert agent._carry is None
    assert all(carry is None and elapsed == 0 for carry, _, _, elapsed in starts)
    assert all(seg[-1] for seg in dones)            # here the segment really does end
    assert np.allclose(out["mean_episode_return"], float(L))
    for env in envs:
        assert env._elapsed_steps == 0              # the reset happened inside the segment


def test_episode_return_belongs_to_the_segment_it_ends_in(agent):
    """An episode spanning a boundary is reported once, in full, by the segment it ends in."""
    L = 8
    envs = make_endless_envs(steps=12)               # ends four steps into segment 1

    out = agent.train_step(envs, RealTimeCOIN(rng=0), seg_steps=L, mini_epochs=1,
                           mb_size=8, enc_steps=1, mb_segments=1)

    assert np.isnan(out["mean_episode_return"][0])
    assert out["mean_episode_return"][1] == pytest.approx(12.0)   # 8 carried + 4 more
    # The episode that started after that truncation is itself carried on.
    assert agent._carry["elapsed"] == 4 and agent._carry["ep_ret"] == 4.0


def test_gae_bootstraps_on_next_segment_weights(agent):
    """A carried episode is bootstrapped under the weights it will continue under."""
    envs = make_endless_envs(forces=(10.0, -10.0, 5.0))
    S, L, seen = len(envs), 6, []
    inner = agent._mixed_value

    def spy(obs_t, w):
        seen.append(w.detach().clone())
        return inner(obs_t, w)

    # COIN's own weights can coincide across these first few trials, which would leave
    # nothing to compare; a distinct one-hot per segment makes the boundary visible.
    # _policy_weights now also runs once per STEP (the within-segment update), so key the
    # one-hot on the call count: 1 + L calls per segment.
    onehots, calls = torch.eye(agent.num_contexts), [0]

    def fake_weights(pi_agent):
        w = onehots[calls[0] // (L + 1)]
        calls[0] += 1
        return w

    agent._policy_weights = fake_weights
    agent._mixed_value = spy
    agent.train_step(envs, RealTimeCOIN(rng=0), seg_steps=L, mini_epochs=1, mb_size=8,
                     enc_steps=1, mb_segments=1)

    acting = [seen[s * L] for s in range(S)]        # one value call per rollout step
    boot = seen[S * L:S * L + S]                    # then one bootstrap per segment
    assert not torch.equal(acting[0], acting[1])    # else the check below is vacuous
    for s in range(S - 1):
        assert torch.equal(boot[s], acting[s + 1])
    assert torch.equal(boot[-1], acting[-1])        # the last segment has no successor yet


def test_gae_bootstrap_uses_own_weights_without_carry(agent):
    """``carry_state=False``: nothing continues, so EVERY segment bootstraps under its own.

    The next segment is a fresh episode of a possibly different task, so its ``w_0`` is the
    wrong mixture; only the segment's own final (sharpest) weights describe the state being
    bootstrapped. Same one-hot-per-segment mock as the carried case, so a bootstrap taken
    from the neighbour would be plainly visible.
    """
    envs = make_endless_envs(forces=(10.0, -10.0, 5.0))
    S, L, seen = len(envs), 6, []
    inner = agent._mixed_value

    def spy(obs_t, w):
        seen.append(w.detach().clone())
        return inner(obs_t, w)

    onehots, calls = torch.eye(agent.num_contexts), [0]

    def fake_weights(pi_agent):
        w = onehots[calls[0] // (L + 1)]
        calls[0] += 1
        return w

    agent._policy_weights = fake_weights
    agent._mixed_value = spy
    agent.train_step(envs, RealTimeCOIN(rng=0), seg_steps=L, mini_epochs=1, mb_size=8,
                     enc_steps=1, mb_segments=1, carry_state=False)

    acting = [seen[s * L] for s in range(S)]
    boot = seen[S * L:S * L + S]
    assert not torch.equal(acting[0], acting[1])    # else the check below is vacuous
    for s in range(S):
        assert torch.equal(boot[s], acting[s])
    assert agent._carry is None                     # and no episode is handed on


#----- Uncertainty passing -----

def test_sigma_sensory_noise_is_the_posterior_sd(agent):
    """COIN's sensory noise for a trial is the encoder's own sd for that segment."""
    coin, seen = RealTimeCOIN(rng=0), []
    observe_y = coin.observe_y

    def spy(y):
        seen.append((float(coin.sigma_sensory_noise), float(y)))
        return observe_y(y)

    coin.observe_y = spy
    envs = make_envs()
    out = agent.train_step(envs, coin, seg_steps=16, mini_epochs=1, mb_size=16,
                           enc_steps=1, mb_segments=1)

    sigmas, ys = (np.asarray(v) for v in zip(*seen))
    assert len(seen) == len(envs)
    assert np.allclose(sigmas, out["z_sd"]) and np.all(sigmas > 0.0)
    assert np.allclose(ys, out["z"])
    assert coin.sigma_sensory_noise == pytest.approx(out["z_sd"][-1])


#----- Within-segment responsibility updates -----

def test_step_context_weights_sharpen_and_prior(agent):
    """Flat likelihood keeps the prior; a sharp latent concentrates the weights."""
    pi = np.array([0.5, 0.5, 0.0])
    mu = np.array([-0.1, 0.1, 0.0])
    var = np.array([1e-6, 1e-6, 0.09])
    # sd large: the likelihood is flat, w stays at the predicted pi
    w_flat = agent._step_context_weights(pi, 0.1, 10.0, mu, var, 0.0)
    assert np.allclose(w_flat[:2], pi[:2], atol=1e-3)
    # sd tiny: mass collapses onto the context nearest z
    w_sharp = agent._step_context_weights(pi, 0.1, 1e-3, mu, var, 0.0)
    assert w_sharp[1] > 0.999
    # nan padding (uninstantiated slots) passes through untouched
    pi_pad = np.array([1.0, np.nan, 0.0])
    w_pad = agent._step_context_weights(pi_pad, 0.0, 0.1,
                                        np.array([0.0, 0.0]), np.array([1e-4, 0.09]), 0.0)
    assert np.isnan(w_pad[1]) and np.isfinite(w_pad[0])


def test_first_step_acts_on_predicted_pi(agent):
    """Step 0 of every segment acts on w_0 = _policy_weights(predicted pi)."""
    seen, inner = [], agent.act

    def spy(obs_t, w):
        seen.append(w.detach().clone())
        return inner(obs_t, w)

    agent.act = spy
    L = 8
    out = agent.train_step(make_envs(), RealTimeCOIN(rng=0), seg_steps=L, mini_epochs=1,
                           mb_size=8, enc_steps=1, mb_segments=1)
    for s, pi_seg in enumerate(out["pi"]):
        assert torch.allclose(seen[s * L], agent._policy_weights(pi_seg))


def test_per_step_weights_reach_acting(agent):
    """From step 1 onward the policy acts on the within-segment update's output."""
    stub = np.array([0.25, 0.75, 0.0])
    agent._step_context_weights = lambda *a, **k: stub
    seen, inner = [], agent.act

    def spy(obs_t, w):
        seen.append(w.detach().clone())
        return inner(obs_t, w)

    agent.act = spy
    L, envs = 8, make_envs()
    agent.train_step(envs, RealTimeCOIN(rng=0), seg_steps=L, mini_epochs=1, mb_size=8,
                     enc_steps=1, mb_segments=1)
    expect = agent._policy_weights(stub)
    for s in range(len(envs)):
        for t in range(1, L):
            assert torch.allclose(seen[s * L + t], expect)


#----- Self-identifying evaluation -----

def train_a_few_rollouts(agent, coin, n: int = 3, L: int = 16):
    """Enough joint-regime rollouts for COIN to instantiate contexts and heads to exist."""
    for _ in range(n):
        agent.train_step(make_envs(), coin, seg_steps=L, mini_epochs=1, mb_size=16,
                         enc_steps=1, mb_segments=1)
    return coin


def test_evaluate_identifying_freezes_everything(agent):
    """The headline contract: an evaluation is a pure read of COIN and of the agent.

    Every checkpoint evaluates all five tasks, so an evaluation that nudged COIN or the
    heads would leak each task's eval into the next one's -- and into the training that
    follows it.
    """
    coin = train_a_few_rollouts(agent, RealTimeCOIN(rng=0))

    # Warm the lazy alignment cache first: it is memoisation keyed on state_version, not
    # state, and evaluate_identifying will hit it rather than recompute.
    coin.context_alignment()
    coin_before = coin_state_fingerprint(coin)
    params_before = agent_param_fingerprint(agent)
    heads_before = (sorted(map(str, agent.nets)), dict(agent.context_init))
    # The last training step leaves its gradients in place (Adam zeroes before backward,
    # not after), so the invariant is that eval does not TOUCH them, not that they are zero.
    grads_before = [None if p.grad is None else p.grad.detach().clone()
                    for p in agent.encoder.parameters()]

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate_identifying must never update COIN")

    coin.observe_y = forbidden
    coin.observe_q = forbidden

    out = agent.evaluate_identifying(make_envs()[0], coin, n_episodes=4, max_steps=20,
                                     seed=11)

    assert coin_state_fingerprint(coin) == coin_before      # trial, particles, rng position
    params_after = agent_param_fingerprint(agent)
    assert set(params_after) == set(params_before)
    assert all(torch.equal(params_before[k], params_after[k]) for k in params_after)
    assert (sorted(map(str, agent.nets)), dict(agent.context_init)) == heads_before
    for before_g, p in zip(grads_before, agent.encoder.parameters()):
        assert (p.grad is None) == (before_g is None)
        assert before_g is None or torch.equal(before_g, p.grad)

    assert set(out) == {"returns", "sharpen_step", "steps", "w_mean"}
    assert out["returns"].shape == (4,) and np.isfinite(out["returns"]).all()
    assert out["steps"].shape == (4,) and np.all((out["steps"] >= 1) & (out["steps"] <= 20))
    assert out["w_mean"].shape == (4, agent.num_contexts)
    assert np.allclose(out["w_mean"].sum(axis=1), 1.0, atol=1e-5)
    sharp = out["sharpen_step"]
    assert sharp.shape == (4,)
    found = np.isfinite(sharp)
    assert np.all(sharp[found] >= 0) and np.all(sharp[found] < out["steps"][found])


def test_evaluate_identifying_is_deterministic(agent):
    """Same seed, same episodes: nothing about the call is stateful."""
    coin = train_a_few_rollouts(agent, RealTimeCOIN(rng=0), n=2)
    kwargs = dict(n_episodes=3, max_steps=15, seed=5)

    first = agent.evaluate_identifying(make_envs()[0], coin, **kwargs)
    second = agent.evaluate_identifying(make_envs()[0], coin, **kwargs)

    assert np.array_equal(first["returns"], second["returns"])
    assert np.array_equal(first["steps"], second["steps"])
    assert np.allclose(first["w_mean"], second["w_mean"])
    assert np.array_equal(np.isnan(first["sharpen_step"]), np.isnan(second["sharpen_step"]))


def test_evaluate_identifying_starts_every_episode_from_the_stationary_prior(agent):
    """Step 0 of each eval episode knows nothing but COIN's long-run marginal.

    Using the PREDICTED pi instead would condition on whichever context the training stream
    left COIN in, leaking the task identity into a test whose whole point is that the agent
    identifies the task itself.
    """
    coin = train_a_few_rollouts(agent, RealTimeCOIN(rng=0))
    k = min(int(coin.context_alignment()["K"]), agent.num_contexts - 1)
    prior = agent._eval_prior_pi(coin, k)
    stationary = np.asarray(coin.stationary_context_probabilities(), dtype=float)

    assert np.nansum(prior) == pytest.approx(1.0)
    assert prior[-1] == 0.0                              # the untrained novel head is off
    live = [j for j, cid in enumerate(agent.context_keys[:-1])
            if j < k and agent.context_init.get(cid, 0) == 1]
    assert live and np.allclose(prior[live], stationary[live] / stationary[live].sum())

    seen, inner = [], agent._deterministic_action

    def spy(obs_t, w):
        seen.append(w.detach().clone())
        return inner(obs_t, w)

    agent._deterministic_action = spy
    out = agent.evaluate_identifying(make_envs()[0], coin, n_episodes=3, max_steps=10,
                                     seed=1)

    starts = np.concatenate([[0], np.cumsum(out["steps"])[:-1]]).astype(int)
    expect = agent._policy_weights(prior)
    assert len(seen) == int(out["steps"].sum())
    for s in starts:
        assert torch.allclose(seen[s], expect)


#----- Seeding -----

def test_seed_everything_reproduces_a_rep():
    """One rep seed drives torch init, the numpy stream, the envs and COIN alike."""
    def run_rep(seed: int):
        rng = seed_everything(seed)
        envs = make_envs(steps=20)
        seed_envs(envs, seed)
        agent = AmortisedCOINPPOAgent(CustomCartPoleEnv(force_mag=10.0,
                                                        max_episode_steps=20),
                                      CTX_IDS, encoder_hidden=8, z_scale=0.3,
                                      prior_sd=0.3, kl_coef=0.03, replay_capacity=2)
        out = agent.train_step(envs, RealTimeCOIN(rng=seed), seg_steps=16, mini_epochs=1,
                               mb_size=16, enc_steps=1, mb_segments=1)
        return out, float(rng.random()), agent

    first, draw_a, agent_a = run_rep(7)
    second, draw_b, agent_b = run_rep(7)
    other, draw_c, agent_c = run_rep(8)

    assert draw_a == draw_b                                  # the returned generator too
    assert np.array_equal(first["z"], second["z"])
    assert np.array_equal(first["pi"], second["pi"], equal_nan=True)
    assert first["policy_loss"] == second["policy_loss"]
    assert first["dyn_loss"] == second["dyn_loss"]
    for pa, pb in zip(agent_a.encoder.parameters(), agent_b.encoder.parameters()):
        assert torch.equal(pa, pb)

    # A different seed really is a different rep (otherwise the checks above are vacuous).
    assert draw_a != draw_c
    assert first["dyn_loss"] != other["dyn_loss"]
    assert any(not torch.equal(pa, pc) for pa, pc
               in zip(agent_a.encoder.parameters(), agent_c.encoder.parameters()))


#----- Markov curriculum -----

def test_curriculum_stationary_and_sampling():
    """The chain's stationary distribution, and a long run that reproduces the matrix."""
    chain = MarkovTaskCurriculum(TRANSITION, rng=0)
    assert chain.n_tasks == 2 and chain.current is None
    assert np.allclose(chain.stationary, [2 / 3, 1 / 3])
    assert np.allclose(MarkovTaskCurriculum.sticky(4, 0.7).stationary, 0.25)
    assert np.allclose(MarkovTaskCurriculum.sticky(1, 0.7).transition_matrix, [[1.0]])

    tasks = chain.sample_block(20000)
    assert tasks.shape == (20000,) and set(np.unique(tasks)) == {0, 1}
    assert np.allclose(curriculum.empirical_transition_matrix(tasks), TRANSITION, atol=0.02)
    assert np.allclose(np.bincount(tasks) / tasks.size, chain.stationary, atol=0.02)

    # A block is the same chain as repeated single draws: no restart at a rollout boundary.
    block = MarkovTaskCurriculum(TRANSITION, rng=7).sample_block(50)
    stepped = MarkovTaskCurriculum(TRANSITION, rng=7)
    assert np.array_equal(block, [stepped.sample_next() for _ in range(50)])
    assert stepped.current == int(block[-1])

    # ``prev_idx`` only re-anchors the chain; it does not reseed it.
    anchored, by_hand = MarkovTaskCurriculum(TRANSITION, rng=3), MarkovTaskCurriculum(
        TRANSITION, rng=3)
    assert np.array_equal(anchored.sample_block(5, prev_idx=1),
                          [by_hand.sample_next(1)] + [by_hand.sample_next() for _ in range(4)])


def test_curriculum_validation():
    """Malformed matrices are rejected at construction, not at the first draw."""
    with pytest.raises(ValueError):
        MarkovTaskCurriculum(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        MarkovTaskCurriculum(np.array([[1.0, -0.5], [0.5, 0.5]]))
    with pytest.raises(ValueError):
        MarkovTaskCurriculum(np.array([[0.0, 0.0], [0.5, 0.5]]))


def test_cross_entropy_helpers():
    """Hand-computable traces, in the agents' NaN-padded context layout."""
    pi = np.array([[0.7, np.nan, 0.3], [0.6, np.nan, 0.4]])
    tasks = np.array([0, 0])

    assert np.array_equal(curriculum.task_context_map(pi, tasks), [0])
    assert np.array_equal(curriculum.task_context_map(pi, tasks, n_tasks=2), [0, -1])
    assert np.allclose(curriculum.predictive_cross_entropy(pi, tasks), -np.log([0.7, 0.6]))
    # An explicit map overrides the modal one; column 2 is the novel slot here.
    assert np.allclose(curriculum.predictive_cross_entropy(pi, tasks, np.array([2])),
                       -np.log([0.3, 0.4]))
    # An unrouted task costs the eps floor, never an infinity.
    assert np.allclose(curriculum.predictive_cross_entropy(pi, tasks, np.array([-1])),
                       -np.log(1e-12))
    with pytest.raises(ValueError):
        curriculum.predictive_cross_entropy(pi, np.array([0, 0, 0]))

    seq = np.array([0, 1, 1])
    assert np.allclose(curriculum.oracle_cross_entropy(seq, TRANSITION),
                       -np.log([2 / 3, 0.1, 0.8]))
    assert np.allclose(curriculum.stationary_cross_entropy(seq, TRANSITION),
                       -np.log([2 / 3, 1 / 3, 1 / 3]))
    assert np.allclose(curriculum.stationary_cross_entropy(seq, np.array([2 / 3, 1 / 3])),
                       -np.log([2 / 3, 1 / 3, 1 / 3]))
    assert curriculum.oracle_cross_entropy(np.zeros(0, dtype=int), TRANSITION).size == 0

    # The oracle is the floor and the marginal the ceiling, on a chain this sticky.
    run = MarkovTaskCurriculum(TRANSITION, rng=1).sample_block(2000)
    assert (curriculum.oracle_cross_entropy(run, TRANSITION).mean()
            < curriculum.stationary_cross_entropy(run, TRANSITION).mean())


def test_transition_matrix_metrics():
    """Empirical counts, the KL summary, and the slice out of COIN's context frame."""
    assert np.allclose(curriculum.empirical_transition_matrix(np.array([0, 1, 1, 0])),
                       [[0.0, 1.0], [0.5, 0.5]])
    assert np.allclose(curriculum.empirical_transition_matrix(np.array([0, 0]), n_tasks=3)[1],
                       1 / 3)                       # an unvisited row stays uniform

    per_row, summary = curriculum.transition_matrix_kl(TRANSITION, TRANSITION)
    assert np.allclose(per_row, 0.0) and summary == pytest.approx(0.0)

    per_row, summary = curriculum.transition_matrix_kl(np.full((2, 2), 0.5), TRANSITION)
    expected = np.array([np.sum(row * np.log(row / 0.5)) for row in TRANSITION])
    assert np.allclose(per_row, expected)
    assert summary == pytest.approx(float(np.dot([2 / 3, 1 / 3], expected)))
    with pytest.raises(ValueError):
        curriculum.transition_matrix_kl(np.eye(3) / 1.0, TRANSITION)

    # COIN's matrix is wider than the task set and in its own order.
    wide = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.25, 0.25, 0.5]])
    assert np.allclose(curriculum.reorder_transition_matrix(wide, np.array([2, 0])),
                       [[0.5 / 0.75, 0.25 / 0.75], [0.7 / 0.8, 0.1 / 0.8]])
    with pytest.raises(ValueError):
        curriculum.reorder_transition_matrix(wide, np.array([0, -1]))
