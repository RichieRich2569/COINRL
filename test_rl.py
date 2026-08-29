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
import numpy as np
import pytest
import torch
import torch.nn as nn

import curriculum
from curriculum import MarkovTaskCurriculum
from environments import CustomCartPoleEnv
from rl import (AmortisedCOINPPOAgent, ContingencyEncoder, SegmentReplayBuffer,
                coin_predicted_pi)

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
    """Feature layout is ``concat(s, one_hot(a), r, s')``."""
    enc = make_encoder()
    T = 6
    obs = torch.randn(T, OBS_DIM)
    next_obs = torch.randn(T, OBS_DIM)
    act = torch.arange(T) % ACT_DIM
    rew = torch.arange(T, dtype=torch.float32)

    feats = enc.transition_features(obs, act, rew, next_obs)

    assert feats.shape == (T, 2 * OBS_DIM + ACT_DIM + 1) == (T, enc.in_dim)
    assert torch.allclose(feats[:, :OBS_DIM], obs)
    onehot = feats[:, OBS_DIM:OBS_DIM + ACT_DIM]
    assert torch.equal(onehot.sum(dim=1), torch.ones(T))
    assert torch.equal(onehot.argmax(dim=1), act)
    assert torch.allclose(feats[:, OBS_DIM + ACT_DIM], rew)
    assert torch.allclose(feats[:, -OBS_DIM:], next_obs)


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
                            "policy_loss", "dyn_loss", "encoder_kl", "enc_grad_norm"}
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

    assert set(history) == {"dyn_loss", "encoder_kl", "enc_grad_norm"}
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
    """FIFO in segments, uniform draws without replacement, detached storage."""
    buf = SegmentReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(torch.full((4, 2), float(i)))

    assert len(buf) == 3
    assert [float(f[0, 0]) for f in buf.buffer] == [2.0, 3.0, 4.0]   # oldest evicted

    assert buf.sample(2).shape == (2 * 4, 2)
    assert buf.sample(10).shape == (3 * 4, 2)                        # capped at the pool
    assert sorted({float(v) for v in buf.sample(3)[:, 0]}) == [2.0, 3.0, 4.0]

    live = torch.zeros(6, 2, requires_grad=True)
    buf.push(live)                          # a new segment length empties the pool
    assert len(buf) == 1 and not buf.buffer[-1].requires_grad

    with pytest.raises(RuntimeError):
        SegmentReplayBuffer().sample(1)


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
    assert agent.replay.sample(2 * S).shape == (2 * S * L, agent.encoder.in_dim)


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
