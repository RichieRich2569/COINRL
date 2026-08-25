"""
test_rl.py

Basic pytest coverage for the dynamics-based amortised COIN-PPO block of :mod:`rl`:
the encoder's prefix posterior and feature layout, the decoder, the property that made
dynamics training worth the switch (a non-zero encoder gradient at initialisation), the
COIN interface helpers, and end-to-end smoke tests of both training regimes.

Run with ``.venv/Scripts/python -m pytest test_rl.py -v``.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from environments import CustomCartPoleEnv
from rl import AmortisedCOINPPOAgent, ContingencyEncoder, coin_predicted_pi

realtimecoin = pytest.importorskip("realtimecoin")
RealTimeCOIN = realtimecoin.RealTimeCOIN

OBS_DIM, ACT_DIM = 4, 2       # CartPole
CTX_IDS = {0: 0, 1: 1}        # -> context_keys [0, 1, "novel"], num_contexts == 3


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

        assert set(out) == {"z", "z_sd", "K", "pi", "rho", "mean_episode_return",
                            "mean_reward_per_step", "value_loss", "policy_loss",
                            "dyn_loss", "encoder_kl", "enc_grad_norm"}
        for key in ("z", "z_sd", "K", "mean_episode_return"):
            assert out[key].shape == (S,)
        for key in ("pi", "rho"):
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

    history = agent.pretrain_encoder(envs, seg_steps=32, n_iters=2, mini_epochs=2, mb_size=16)

    assert set(history) == {"dyn_loss", "encoder_kl", "enc_grad_norm"}
    for values in history.values():
        assert values.shape == (2,) and np.isfinite(values).all()
    assert any(not torch.equal(b, p)
               for b, p in zip(before, agent.encoder.parameters()))

    frozen = [p.detach().clone() for p in agent.encoder.parameters()]
    out = agent.train_step(make_envs(), RealTimeCOIN(rng=0), seg_steps=32, mini_epochs=2,
                           mb_size=16, update_encoder=False)

    assert out["enc_grad_norm"] == 0.0
    assert np.isfinite(out["dyn_loss"]) and np.isfinite(out["encoder_kl"])
    for f, p in zip(frozen, agent.encoder.parameters()):
        assert torch.equal(f, p)
