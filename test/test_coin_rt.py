# test_coin_rt_real.py
from __future__ import annotations

import importlib
import numpy as np
import pytest


def import_modules():
    """
    Import the real modules. If `coin` or your COIN_RT module isn't importable,
    we skip the suite to avoid false negatives on CI.
    """
    try:
        coin = importlib.import_module("coin")
    except Exception as e:
        pytest.skip(f"Real 'coin' module not importable: {e}")

    # Change 'coin_rt' to your actual module filename (without .py) if needed
    try:
        if "coin_rt" in globals():
            importlib.reload(coin_rt)  # type: ignore # if re-running locally
        coin_rt = importlib.import_module("coin_rt")
    except Exception as e:
        pytest.skip(f"'coin_rt' module not importable: {e}")

    # sanity check the class exists
    assert hasattr(coin_rt, "COIN_RT"), "COIN_RT class not found in coin_rt module"
    return coin, coin_rt


def test_defaults_initialise_without_cues():
    _, coin_rt = import_modules()
    model = coin_rt.COIN_RT()  # defaults

    # Basic invariants from COIN_RT
    assert model.trial == 0
    assert model.max_cues == 10
    assert isinstance(model.coin_state, dict)
    assert model.perturbations == []
    assert model.cues is None

    # With cues_exist=False, cue-specific keys should NOT be present
    assert model.coin_state["cues_exist"] == 0
    assert "Q" not in model.coin_state
    assert "n_cue" not in model.coin_state

    # If the real base class exposes particles/max_contexts, check they match defaults
    if hasattr(model, "particles"):
        assert model.particles == 100
    if hasattr(model, "max_contexts"):
        assert model.max_contexts == 10


@pytest.mark.parametrize(
    "particles,max_contexts,max_cues,trial",
    [
        (64, 7, 5, 3),
        (8, 2, 2, 11),
    ],
)
def test_initialise_with_cues_shapes_and_fields(particles, max_contexts, max_cues, trial):
    _, coin_rt = import_modules()
    model = coin_rt.COIN_RT(
        particles=particles,
        max_contexts=max_contexts,
        max_cues=max_cues,
        trial=trial,
        cues_exist=True,
    )

    cs = model.coin_state
    # COIN_RT is responsible for these fields when cues_exist=True
    assert cs.get("cues_exist") == 1
    assert cs.get("Q") == 0
    assert "n_cue" in cs

    n_cue = cs["n_cue"]
    assert isinstance(n_cue, np.ndarray)
    assert n_cue.shape == (max_contexts + 1, max_cues + 1, particles)
    assert np.all(n_cue == 0)

    # Attributes persisted
    assert model.trial == trial
    assert model.max_cues == max_cues

    # If base exposes these, they should reflect overrides
    if hasattr(model, "particles"):
        assert model.particles == particles
    if hasattr(model, "max_contexts"):
        assert model.max_contexts == max_contexts


def test_reinitialise_toggles_cue_fields_and_resets_bookkeeping():
    _, coin_rt = import_modules()
    model = coin_rt.COIN_RT(particles=8, max_contexts=2, max_cues=2)

    # Start with default (cues_exist=False)
    assert model.coin_state["cues_exist"] == 0
    assert model.perturbations == []
    assert model.cues is None

    # Mutate bookkeeping to confirm reset happens
    model.perturbations = [1, 2, 3]
    model.cues = ["x"]

    # Re-initialise with cues_exist=True
    cs2 = model.initialise_coin(cues_exist=True)
    assert cs2["cues_exist"] == 1
    assert cs2["Q"] == 0
    assert "n_cue" in cs2
    assert cs2["n_cue"].shape == (model.max_contexts + 1, model.max_cues + 1, model.particles)

    # Bookkeeping reset by initialise_coin
    assert model.perturbations == []
    assert model.cues == [2] # cues list is empty list + maximum number of cues

    # Re-initialise again with cues_exist=False (cue fields should disappear)
    cs3 = model.initialise_coin(cues_exist=False)
    assert cs3["cues_exist"] == 0
    assert "Q" not in cs3
    assert "n_cue" not in cs3


def test_parameter_overrides_do_not_crash_and_are_reflected_if_exposed():
    _, coin_rt = import_modules()
    model = coin_rt.COIN_RT(
        sigma_process_noise=0.01,
        sigma_sensory_noise=0.05,
        prior_mean_retention=0.9,
        prior_precision_retention=100.0,
        prior_precision_drift=400.0,
        gamma_context=0.2,
        alpha_context=5.0,
        rho_context=0.3,
        gamma_cue=0.4,
        alpha_cue=12.0,
        infer_bias=True,
        prior_precision_bias=50.0,
        runs=2,
        max_cores=4,
        particles=32,
        max_contexts=3,
        max_cues=4,
        trial=42,
        cues_exist=True,
    )

    # Sanity: model constructed and cue fields present
    assert model.coin_state["cues_exist"] == 1
    assert model.coin_state["Q"] == 0
    assert model.coin_state["n_cue"].shape == (3 + 1, 4 + 1, 32)

    # Only assert on attributes if the real base exposes them
    if hasattr(model, "particles"):
        assert model.particles == 32
    if hasattr(model, "max_contexts"):
        assert model.max_contexts == 3

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        pytest.skip(f"Module '{module_name}' not importable: {e}")


def import_modules():
    coin = _try_import("coin")
    coin_rt = _try_import("coin_rt")  # adjust if your file name differs
    assert hasattr(coin_rt, "COIN_RT"), "COIN_RT class not found in coin_rt module"
    return coin, coin_rt


def _ensure_list_in_state(coin_state: dict, key: str):
    if key not in coin_state or coin_state[key] is None:
        coin_state[key] = []


def _rng(seed: int = 123):
    rs = np.random.RandomState(seed)
    return rs


# ---------------------------------------------------------------------
# Fixtures: models with/without cues
# ---------------------------------------------------------------------

@pytest.fixture
def model_no_cues():
    _, coin_rt = import_modules()
    m = coin_rt.COIN_RT(cues_exist=False)
    # Ensure fields required by step() exist if base doesn't set them
    m.coin_state.setdefault("trial", 0)
    _ensure_list_in_state(m.coin_state, "feedback_observed")
    return m


@pytest.fixture
def model_with_cues():
    _, coin_rt = import_modules()
    m = coin_rt.COIN_RT(cues_exist=True, max_contexts=4, max_cues=3, particles=10)
    # Minimal fields for methods that expect them (if base doesn't provide)
    m.coin_state.setdefault("trial", 0)
    _ensure_list_in_state(m.coin_state, "feedback_observed")
    # Ensure cues container; current implementation sets `self.cues=None`,
    # but step() calls `self.cues.insert(...)`. We *expect* a list.
    # We do NOT fix it here so we can surface the bug in a dedicated test.
    return m


# ---------------------------------------------------------------------
# 1) Validation & bookkeeping in `step`
# ---------------------------------------------------------------------

def test_step_raises_when_cues_required_but_missing(model_with_cues):
    """
    When cues_exist=True and cue=None, step() must raise ValueError *after* incrementing trial.
    """
    m = model_with_cues
    # Ensure cues list exists so error comes specifically from cue validation (not AttributeError)
    m.cues = []
    trial_before = m.coin_state["trial"]
    with pytest.raises(ValueError, match="Cue must be provided when cues_exist is True."):
        m.step(state_feedback=0.1, cue=None)
    assert m.coin_state["trial"] == trial_before + 1, "trial should increment even if validation fails"


def test_step_warns_when_cue_provided_but_cues_do_not_exist(model_no_cues, capsys, monkeypatch):
    """
    When cues_exist=False and cue is provided, a warning is printed and cue ignored.
    We monkeypatch the heavy methods to no-ops so we can run `step`.
    """
    m = model_no_cues

    # Minimal coin_state scaffolding for the call chain (if base doesn't provide)
    m.coin_state.setdefault("prior_probabilities", np.ones((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("predicted_probabilities", np.ones((m.max_contexts + 1, m.particles)) / (m.max_contexts + 1))
    m.coin_state.setdefault("state_mean", np.zeros((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("bias", np.zeros((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("state_var", np.ones((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("sigma_observation_noise", np.ones((m.max_contexts + 1, m.particles)) * 0.1)
    m.coin_state.setdefault("average_state", 0.0)

    # Monkeypatch heavy calls to simple pass-throughs
    monkeypatch.setattr(m, "predict_context", lambda cs, cue=None: cs, raising=True)
    monkeypatch.setattr(m, "predict_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "resample_particles", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "sample_context", lambda cs, cue=None: cs, raising=True)
    monkeypatch.setattr(m, "update_belief_about_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "sample_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_for_parameters", lambda cs, cue=None: cs, raising=True)
    monkeypatch.setattr(m, "sample_parameters", lambda cs: cs, raising=True)
    # `predict_state_feedback` is defined in child; let it run, but stub compute_marginal_distribution
    monkeypatch.setattr(m, "compute_marginal_distribution", lambda cs: cs, raising=True)

    def _store_stub(cs):
        cs["stored"] = {n: {"ok": True} for n in range(m.runs)}
        return cs

    monkeypatch.setattr(m, "store_variables", _store_stub, raising=True)

    # Run with cue present but cues_exist=False
    res = m.step(state_feedback=0.2, cue=1)
    out = capsys.readouterr().out
    assert "Warning: Cue provided but cues_exist is False" in out
    assert isinstance(res, dict) and "runs" in res and "weights" in res and "properties" in res
    assert set(res["runs"].keys()) == set(range(m.runs))


# ---------------------------------------------------------------------
# 2) predict_context (with/without cues) - shapes and normalization
# ---------------------------------------------------------------------

def test_predict_context_without_cues_direct_call(monkeypatch, model_with_cues):
    """
    Directly test predict_context with synthetic transition probabilities and no cues.
    """
    m = model_with_cues
    cs = dict(
        cues_exist=0,
        trial=1,
        context=np.ones((m.particles,), dtype=int),  # all in context 1
    )
    # local_transition_matrix[from, to, particle]
    rs = _rng(7)
    ltm = rs.rand(m.max_contexts + 1, m.max_contexts + 1, m.particles)
    cs["local_transition_matrix"] = ltm

    # Do not trigger optional store-dependent branches
    m.store = []

    out = m.predict_context(cs, cue=None)
    assert "prior_probabilities" in out and "predicted_probabilities" in out
    assert out["prior_probabilities"].shape == (m.max_contexts + 1, m.particles)
    assert out["predicted_probabilities"].shape == (m.max_contexts + 1, m.particles)
    # With no cues, predicted_probabilities == prior_probabilities (no normalization forced)
    np.testing.assert_allclose(out["predicted_probabilities"], out["prior_probabilities"])


def test_predict_context_with_cues_normalizes_columns(monkeypatch, model_with_cues):
    """
    With cues_exist=True and a cue provided, predicted_probabilities should be column-normalized.
    """
    m = model_with_cues
    cs = dict(
        cues_exist=1,
        trial=2,
        context=np.ones((m.particles,), dtype=int),
        Q=0,
    )
    rs = _rng(9)
    cs["local_transition_matrix"] = rs.rand(m.max_contexts + 1, m.max_contexts + 1, m.particles)
    cs["local_cue_matrix"] = rs.rand(m.max_contexts + 1, m.max_cues + 1, m.particles)

    m.store = []  # avoid optional branches

    out = m.predict_context(cs, cue=1)
    assert out["predicted_probabilities"].shape == (m.max_contexts + 1, m.particles)
    col_sums = np.sum(out["predicted_probabilities"], axis=0, keepdims=True)
    np.testing.assert_allclose(col_sums, np.ones_like(col_sums), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 3) predict_state_feedback computations
# ---------------------------------------------------------------------

def test_predict_state_feedback_means_vars_and_motor_output(model_with_cues, monkeypatch):
    """
    Verify:
    - state_feedback_mean = state_mean + bias
    - state_feedback_var = state_var + sigma_observation_noise^2
    - motor_output = mean over particles of sum(predicted_probabilities * state_feedback_mean)
    - prediction_error = y_t - state_feedback_mean (per-context, per-particle)
    """
    m = model_with_cues
    m.store = ["implicit"]  # also test `implicit = motor_output - average_state`

    C = m.max_contexts + 1
    P = m.particles

    cs = dict(
        cues_exist=1,
        trial=1,
        predicted_probabilities=np.ones((C, P)) / C,
        state_mean=np.full((C, P), 0.3),
        bias=np.full((C, P), 0.1),
        state_var=np.full((C, P), 0.04),  # variance
        sigma_observation_noise=np.full((C, P), 0.2),
        average_state=0.25,
    )

    # Avoid dependence on base method
    monkeypatch.setattr(m, "compute_marginal_distribution", lambda d: d, raising=True)

    y = 0.8
    out = m.predict_state_feedback(cs, state_feedback=y)

    expected_mean = 0.3 + 0.1
    expected_var = 0.04 + (0.2 ** 2)
    assert np.allclose(out["state_feedback_mean"], expected_mean)
    assert np.allclose(out["state_feedback_var"], expected_var)

    # motor_output: average over particles of (sum over contexts of P(c)*mean)
    mo = np.sum(out["predicted_probabilities"] * out["state_feedback_mean"]) / P
    assert np.isclose(out["motor_output"], mo)

    # implicit = motor_output - average_state
    assert np.isclose(out["implicit"], out["motor_output"] - 0.25)

    # prediction_error: per context/particle
    assert out["prediction_error"].shape == (C, P)
    np.testing.assert_allclose(out["prediction_error"], y - expected_mean)


# ---------------------------------------------------------------------
# 4) sample_context state updates
# ---------------------------------------------------------------------

def test_sample_context_updates_context_and_stick_breaking(model_with_cues, monkeypatch):
    """
    We build a minimal responsibilities matrix and ensure:
    - context is sampled within [1, C_max] U Cmax+1 - where Cmax+1 is the "novel" context
    - C is incremented where new contexts are sampled
    - global_transition_probabilities rows get updated for new contexts
    """
    m = model_with_cues
    Cmax = m.max_contexts
    P = m.particles

    rs = _rng(42)

    # Set responsibilities and C - responsibilities should be zero for contexts larger than C+1
    resp = np.zeros((Cmax + 1, P))
    resp[0:3, :] = 1.0 / 3.0  # only contexts 0,1,2 have non-zero resp

    # Set gtp - has to sum to one in axis=0 (over contexts)
    gtp = np.zeros((Cmax + 1, P))
    gtp[0:3, :] = rs.rand(3, P)
    gtp /= np.sum(gtp, axis=0, keepdims=True)

    cs = dict(
        cues_exist=0,  # set to 0 to avoid cue validation
        trial=2,
        responsibilities=resp,
        context=np.ones((P,), dtype=int),
        C=np.ones((P,), dtype=int) + 1,  # current max context index per particle (2)
        global_transition_probabilities=gtp,
    )

    # Seed randomness to make behavior deterministic(ish)
    np.random.seed(0)
    out = m.sample_context(cs, cue=None)

    # Contexts should be in 1..Cmax (inclusive)
    assert np.all((out["context"] >= 1) & (out["context"] <= Cmax + 1))
    assert "p_new_x" in out and "p_old_x" in out
    # Where new contexts were sampled, C should have been incremented
    if len(out["p_new_x"]) > 0:
        assert np.all(out["C"][out["p_new_x"]] >= 2)

    # Stick-breaking update touched transition probabilities for some inds
    # (Cannot assert exact values, but we can assert non-negativity and some non-zeros)
    gtp = out["global_transition_probabilities"]
    assert gtp.shape == (Cmax + 1, P)
    assert np.all(gtp >= 0.0)
    assert np.any(gtp > 0.0)
    assert np.all(np.abs(np.sum(gtp, axis=0) - 1.0) < 1e-5) # context probabilities sum to one


# ---------------------------------------------------------------------
# 5) update_sufficient_statistics_for_parameters dispatch logic
# ---------------------------------------------------------------------

def test_update_sufficient_statistics_for_parameters_calls(monkeypatch, model_with_cues):
    """
    We monkeypatch the called methods to record whether they are invoked under the
    documented conditions:
      - always calls global transition stats
      - calls global cue stats only if cues_exist and cue is not None
      - calls dynamics only if trial > 1
      - calls bias only if infer_bias=True and feedback observed
    """
    m = model_with_cues
    m.infer_bias = True

    flags = {
        "gtp": 0,
        "gcp": 0,
        "dyn": 0,
        "bias": 0,
    }

    def _gtp(cs):
        flags["gtp"] += 1
        return cs

    def _gcp(cs, cue):
        assert cue == 1
        flags["gcp"] += 1
        return cs

    def _dyn(cs):
        flags["dyn"] += 1
        return cs

    def _bias(cs):
        flags["bias"] += 1
        return cs

    monkeypatch.setattr(m, "update_sufficient_statistics_global_transition_probabilities", _gtp, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_global_cue_probabilities", _gcp, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_dynamics", _dyn, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_bias", _bias, raising=True)

    cs = dict(
        cues_exist=1,
        trial=2,  # >1 triggers dynamics
        feedback_observed=[True],  # index trial-1 == 1 -> we'll ensure True
    )
    cs["feedback_observed"].append(True)  # so feedback_observed[1] is True

    out = m.update_sufficient_statistics_for_parameters(cs, cue=1)
    assert out is cs
    assert flags["gtp"] == 1
    assert flags["gcp"] == 1
    assert flags["dyn"] == 1
    assert flags["bias"] == 1


# ---------------------------------------------------------------------
# 6) update_sufficient_statistics_global_cue_probabilities increments counts
# ---------------------------------------------------------------------

def test_update_sufficient_statistics_global_cue_probabilities_increments_counts(model_with_cues):
    m = model_with_cues
    P = m.particles
    Cmax = m.max_contexts
    Qmax = m.max_cues

    cs = dict(
        context=np.full((P,), 2, dtype=int),  # context idx 2
        n_cue=np.zeros((Cmax + 1, Qmax, P), dtype=int),
    )
    cue = 1
    out = m.update_sufficient_statistics_global_cue_probabilities(cs, cue=cue)

    # Expect increments at (context-1, cue, particle_idx) for all particles
    for i in range(P):
        assert out["n_cue"][1, cue, i] == 1  # context-1 == 1


# ---------------------------------------------------------------------
# 7) End-to-end `step` smoke with minimal monkeypatching
# ---------------------------------------------------------------------

def test_step_end_to_end_smoke_with_cues(monkeypatch, model_with_cues):
    """
    Run several `step` calls with cues_exist=True and sequential cues.
    We stub heavy methods but let child code paths execute.
    """
    m = model_with_cues
    m.runs = 3
    # Ensure cues list exists (current implementation sets to max_cues)
    m.cues = [m.max_cues]

    C = m.max_contexts + 1
    P = m.particles

    # Minimal, consistent state used by child methods
    m.coin_state.setdefault("prior_probabilities", np.ones((C, P)) / C)
    m.coin_state.setdefault("predicted_probabilities", np.ones((C, P)) / C)
    m.coin_state.setdefault("state_mean", np.zeros((C, P)))
    m.coin_state.setdefault("bias", np.zeros((C, P)))
    m.coin_state.setdefault("state_var", np.ones((C, P)) * 0.05)
    m.coin_state.setdefault("sigma_observation_noise", np.ones((C, P)) * 0.1)
    m.coin_state.setdefault("average_state", 0.0)
    m.coin_state.setdefault("context", np.ones((P,), dtype=int))
    m.coin_state.setdefault("C", np.ones((P,), dtype=int))
    m.coin_state.setdefault("global_transition_probabilities", np.zeros((C, P)))
    m.coin_state.setdefault("Q", 0)
    m.coin_state.setdefault("global_cue_probabilities", np.zeros((m.max_cues + 2, P)))  # +2 to allow Q+1 indexing

    # Predict context -> keep as-is but ensure fields exist
    def _predict_context(cs, cue=None):
        # Make a simple prior and (if cue) re-normalize
        cs["prior_probabilities"] = np.ones((C, P)) / C
        if cs.get("cues_exist") and cue is not None:
            cs["predicted_probabilities"] = np.ones((C, P)) / C
        else:
            cs["predicted_probabilities"] = cs["prior_probabilities"]
        return cs

    monkeypatch.setattr(m, "predict_context", _predict_context, raising=True)
    monkeypatch.setattr(m, "predict_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "resample_particles", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "update_belief_about_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "sample_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "sample_parameters", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "compute_marginal_distribution", lambda cs: {**cs, "responsibilities": cs["predicted_probabilities"]}, raising=True)

    def _sample_context(cs, cue=None):
        # Keep context at 1 for simplicity; update Q if introducing new cue
        cs["context"] = np.ones((P,), dtype=int)
        cs["p_new_x"] = np.array([], dtype=int)
        cs["p_old_x"] = np.arange(P)
        if cs.get("cues_exist") and (cue is not None) and (cue > cs.get("Q", 0)):
            cs["Q"] = cs.get("Q", 0) + 1
        return cs

    monkeypatch.setattr(m, "sample_context", _sample_context, raising=True)

    def _update_ss(cs, cue=None):
        return cs

    monkeypatch.setattr(m, "update_sufficient_statistics_for_parameters", _update_ss, raising=True)

    def _store(cs):
        cs["stored"] = {n: {"trial": cs["trial"]} for n in range(m.runs)}
        return cs

    monkeypatch.setattr(m, "store_variables", _store, raising=True)

    # Run three steps with sequential cues (1, then optional repeat)
    res1 = m.step(state_feedback=0.0, cue=1)
    res2 = m.step(state_feedback=np.nan, cue=1)
    res3 = m.step(state_feedback=0.3, cue=2)  # introduce a new cue

    for res in [res1, res2, res3]:
        assert "runs" in res and "weights" in res and "properties" in res
        assert set(res["runs"].keys()) == set(range(m.runs))
        assert res["weights"].shape == (m.runs,)

    # feedback_observed handling
    assert np.all(m.coin_state["feedback_observed"][-3:] == [True, False, True])
    # Q should have incremented to 2
    assert m.coin_state["Q"] == 2
    # cues list should have been filled by `.insert`
    assert np.all(m.cues[:3] == [1, 1, 2])


# ---------------------------------------------------------------------
# 8) Document current behavior: self.cues should be list when cues_exist=True
# ---------------------------------------------------------------------

@pytest.mark.xfail(reason="COIN_RT.initialise_coin sets self.cues=None even when cues_exist=True; step() expects a list.")
def test_initial_cues_container_is_a_list_when_cues_exist(model_with_cues):
    """
    This test documents a likely bug:
      - initialise_coin(cues_exist=True) sets self.cues to None
      - step() calls self.cues.insert(...), which requires a list
    Expected: self.cues should be [] when cues_exist=True.
    """
    m = model_with_cues
    assert isinstance(m.cues, list), "Expected self.cues to be a list when cues_exist=True"


# ---------------------------------------------------------------------
# 9) End-to-end equivalence with batch simulation given same feedback
# ---------------------------------------------------------------------
    
def test_rt_matches_batch_simulation_single_step():
    """
    Verify COIN_RT.step produces identical motor_output as coin.COIN.simulate_coin
    under identical feedback, seeds, and hyperparameters for a single step.
    """

    coin, coin_rt = import_modules()

    common_kwargs = dict(
        particles=5,
        max_contexts=10,
    )

    if not hasattr(coin.COIN, "simulate_coin"):
        pytest.skip("coin.COIN.simulate_coin() not available in this environment.")

    SEED = 12345

    np.random.seed(SEED)
    rt_model = coin_rt.COIN_RT(**common_kwargs, cues_exist=False)
    
    # Simulation directly from coin.coin_main_loop
    np.random.seed(SEED)
    coin_model = coin.COIN(**common_kwargs)
    coin_model.perturbations = np.zeros((1, ))
    trials = np.arange(0, 1)
    out_state = coin_model.coin_main_loop(trials)

    y_out = np.asarray(out_state["state_feedback"])
    mo_batch = np.asarray(out_state["motor_output"])

    # --- Realtime simulation ---

    rt_model.coin_state.setdefault("trial", 0)
    if "feedback_observed" not in rt_model.coin_state:
        rt_model.coin_state["feedback_observed"] = []

    y_in = np.nan if np.isnan(y_out) else float(y_out)
    last_res = rt_model.step(state_feedback=y_in, cue=None)

    mo_rt = np.asarray(last_res["runs"][0]["motor_output"])

    # --- Compare outputs ---
    np.testing.assert_allclose(mo_rt, mo_batch, rtol=0, atol=1e-12)

def test_rt_matches_batch_simulation_multiple_steps():
    """
    Verify COIN_RT.step produces identical motor_output as coin.COIN.simulate_coin
    under identical feedback, seeds, and hyperparameters for multiple steps.
    """

    coin, coin_rt = import_modules()

    common_kwargs = dict(
        particles=5,
        max_contexts=10,
    )

    if not hasattr(coin.COIN, "simulate_coin"):
        pytest.skip("coin.COIN.simulate_coin() not available in this environment.")

    SEED = 12345

    np.random.seed(SEED)
    rt_model = coin_rt.COIN_RT(**common_kwargs, cues_exist=False)
    
    # Simulation directly from coin.coin_main_loop
    np.random.seed(SEED)
    coin_model = coin.COIN(**common_kwargs)
    coin_model.perturbations = np.zeros((5, ))
    y_out = np.zeros((5, ))
    mo_batch = np.zeros((5, ))
    trials = np.arange(0, 5)

    for t in trials:
       out_state = coin_model.coin_main_loop(np.array([t]), coin_state=out_state if t > 0 else None)
       y_out[t] = np.asarray(out_state["state_feedback"])
       mo_batch[t] = np.asarray(out_state["motor_output"])

    # --- Realtime simulation ---

    rt_model.coin_state.setdefault("trial", 0)
    if "feedback_observed" not in rt_model.coin_state:
        rt_model.coin_state["feedback_observed"] = []

    for t in range(5):
        yt = y_out[t]
        y_in = np.nan if np.isnan(yt) else float(yt)
        last_res = rt_model.step(state_feedback=y_in, cue=None)

    mo_rt = np.asarray(last_res["runs"][0]["motor_output"])

    # --- Compare outputs ---
    np.testing.assert_allclose(mo_rt, mo_batch, rtol=0, atol=1e-12)


def _make_perturbations(kind: str):
    """Utility to generate perturbation schedules."""
    if kind == "single":
        # Very short, one step schedule
        return np.zeros((1,))
    elif kind == "medium":
        # Moderate schedule with a few dozen trials
        return np.concatenate([
            np.zeros(5),
            np.ones(5),
            -np.ones(3),
            np.ones(5) * np.nan,
        ]).astype(float)
    elif kind == "complex":
        # Long schedule
        return np.concatenate([
            np.zeros((50,)),        # 50 null
            np.ones((125,)),        # 125 P+
            -np.ones((15,)),        # 15 P-
            np.ones((150,)) * np.nan,  # 150 channel trials
        ]).astype(float)
    else:
        raise ValueError(f"Unknown perturbation kind {kind}")

@pytest.mark.parametrize(
    "perturb_kind",
    ["single", "medium", pytest.param("complex", marks=pytest.mark.slow)]
)
def test_rt_matches_batch_simulation_given_same_feedback(perturb_kind):
    """
    Verify COIN_RT.step produces identical motor_output as coin.COIN.simulate_coin
    under identical feedback, seeds, and hyperparameters for a variety of perturbation schedules.
    This uses the full external view of both models.
    """

    coin, coin_rt = import_modules()

    common_kwargs = dict(
        particles=5,
        max_contexts=10,
        max_cores=0,
    )

    perturbations = _make_perturbations(perturb_kind)
    T = perturbations.shape[0]

    if not hasattr(coin.COIN, "simulate_coin"):
        pytest.skip("coin.COIN.simulate_coin() not available in this environment.")

    SEED = 12345
    np.random.seed(SEED)

    coin_model = coin.COIN(**common_kwargs)
    # Provide the perturbation schedule (true states x*, from which y is drawn)
    coin_model.perturbations = perturbations
    # If your implementation uses cues, ensure consistent absence of cues:
    # coin_model.cues = None

    out_batch = coin_model.simulate_coin()
    assert "runs" in out_batch and 0 in out_batch["runs"]
    assert "state_feedback" in out_batch["runs"][0]
    assert "motor_output" in out_batch["runs"][0]

    y_seq = np.asarray(out_batch["runs"][0]["state_feedback"])
    mo_batch = np.asarray(out_batch["runs"][0]["motor_output"])
    assert y_seq.shape[0] == T
    assert mo_batch.shape[0] == T

    # 2) Realtime simulation with COIN_RT using the *same* feedback
    np.random.seed(SEED)

    rt_model = coin_rt.COIN_RT(**common_kwargs, cues_exist=False)  # no cues in this comparison

    # Ensure minimal bookkeeping fields exist for safety
    rt_model.coin_state.setdefault("trial", 0)
    if "feedback_observed" not in rt_model.coin_state:
        rt_model.coin_state["feedback_observed"] = []

    last_res = None
    for t in range(T):
        y = y_seq[t]
        # Important: COIN_RT.step checks `state_feedback is np.nan` (identity check).
        # Pass the *np.nan singleton* explicitly whenever the value is NaN.
        if np.isnan(y):
            y_in = np.nan
        else:
            y_in = float(y)
        last_res = rt_model.step(state_feedback=y_in, cue=None)

    assert last_res is not None and "runs" in last_res and 0 in last_res["runs"]
    assert "motor_output" in last_res["runs"][0]

    mo_rt = np.asarray(last_res["runs"][0]["motor_output"])
    assert mo_rt.shape[0] == T

    # 3) Compare the two motor_output sequences
    # Use tight tolerances (these should be identical given seed and same observations)
    np.testing.assert_allclose(mo_rt, mo_batch, rtol=0, atol=1e-12)

# =====================================================================
# Structured consistency tests: COIN_RT overridden methods vs COIN
# Ensures child outputs match super() outputs under identical state/seed
# Paste this block at the END of test_coin_rt.py
# =====================================================================

import numpy as np
import pytest

# --- Fixtures & helpers --------------------------------------------------------

@pytest.fixture
def modules():
    """
    Reuse your existing import helper if present; otherwise fall back.
    Must yield (coin_module, coin_rt_module).
    """
    try:
        # Prefer the project's helper if defined above in this file:
        coin, coin_rt = import_modules()  # noqa: F821  (defined earlier in your file)
    except NameError:
        import importlib
        coin = importlib.import_module("coin")
        coin_rt = importlib.import_module("coin_rt")
    return coin, coin_rt


@pytest.fixture
def fresh_models(modules):
    """
    Construct matched parent/child models with the same sizes and neutral hooks.
    """
    coin, coin_rt = modules
    parent = coin.COIN(particles=7, max_contexts=5)          # small but nontrivial sizes
    child  = coin_rt.COIN_RT(particles=7, max_contexts=5, max_cues=4)
    # avoid any special side effects from store hooks unless explicitly set
    parent.store = []
    child.store = []
    return parent, child


@pytest.fixture
def seeded():
    """
    Deterministic numpy + RandomState for generating the same inputs every time.
    Usage: rs = seeded(123)
    """
    def _mk(seed=123):
        np.random.seed(seed)
        try:
            # If your file defines _rng helper earlier, use it for parity with other tests
            rs = _rng(seed)  # noqa: F821
        except NameError:
            rs = np.random.RandomState(seed)
        return rs
    return _mk


def _dcopy(d: dict) -> dict:
    """Shallow dict copy with np.ndarray copied by value."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


# --- Parity tests for overridden methods ---------------------------------------
# Overridden in COIN_RT (based on coin_rt.py): 
#   __init__, initialise_coin, step, predict_context, predict_state_feedback,
#   sample_context, update_sufficient_statistics_for_parameters,
#   update_sufficient_statistics_global_cue_probabilities, store_function
#
# We test parity for all logic-returning ones (not constructor), and skip store_function
# (it’s purely a side-effect hook) and __init__.


# 1) initialise_coin parity (both cues_exist=False and True)
def test_initialise_coin_parity(fresh_models):
    parent, child = fresh_models

    # ---- cues_exist = False
    np.random.seed(0)
    parent.perturbations = np.zeros((4,))
    cs_p0 = parent.initialise_coin() # Initialisation cannot happen without perturbations
    np.random.seed(0)
    cs_c0 = child.initialise_coin(cues_exist=False)

    for k in ("trial", "context", "C"):
        assert k in cs_p0 and k in cs_c0
        assert np.all(cs_p0[k] == cs_c0[k])

    # ---- cues_exist = True
    np.random.seed(1)
    parent.cues = np.arange(1, 5, dtype=int) # We now need cues for initialisation
    cs_p1 = parent.initialise_coin()
    np.random.seed(1)
    cs_c1 = child.initialise_coin(cues_exist=True)

    for k in ("trial", "context", "C", "Q"):
        assert k in cs_p1 and k in cs_c1
        assert np.all(cs_p1[k] == cs_c1[k])


# 2) predict_context parity (no cues branch)
def test_predict_context_parity_no_cues(fresh_models, seeded):
    parent, child = fresh_models
    rs = seeded(2025)
    C = parent.max_contexts + 1
    P = parent.particles

    coin_state = dict(
        cues_exist=0,
        trial=1,
        context=rs.randint(1, C, size=(P,), dtype=int),
        local_transition_matrix=rs.rand(C, C, P),
    )

    out_p = parent.predict_context(_dcopy(coin_state))
    out_c = child.predict_context(_dcopy(coin_state), cue=None)

    # In the no-cue path, child's predicted == parent's prior (same math)
    np.testing.assert_allclose(out_c["prior_probabilities"], out_p["prior_probabilities"])
    np.testing.assert_allclose(out_c["predicted_probabilities"], out_p["prior_probabilities"])


# 3) predict_context parity (with cue branch)
def test_predict_context_parity_with_cue(fresh_models, seeded):
    parent, child = fresh_models
    rs = seeded(7)
    C = parent.max_contexts + 1
    P = parent.particles
    Qmax = child.max_cues + 1
    cue = 2  # any valid cue label (1..Qmax-1)

    coin_state = dict(
        cues_exist=1,
        trial=1,
        context=rs.randint(1, C, size=(P,), dtype=int),
        local_transition_matrix=rs.rand(C, C, P),
        local_cue_matrix=rs.rand(C, Qmax, P),
    )

    # Parent obtains cue via self.cues[trial-1], so set it
    parent.cues = np.array([cue], dtype=int)

    out_p = parent.predict_context(_dcopy(coin_state))
    out_c = child.predict_context(_dcopy(coin_state), cue=cue)

    for k in ("prior_probabilities", "cue_probabilities", "predicted_probabilities"):
        assert k in out_p and k in out_c
        np.testing.assert_allclose(out_c[k], out_p[k])


# 4) predict_state_feedback parity (shared, noise-free quantities)
def test_predict_state_feedback_parity(fresh_models):
    parent, child = fresh_models
    C = parent.max_contexts + 1
    P = parent.particles

    parent.perturbations = np.zeros((3,)) # we need this defined for the parent

    # ensure both compute 'implicit' if that path is enabled
    parent.store = ["implicit"]
    child.store = ["implicit"]

    coin_state = dict(
        trial = 1, # needed for parent predict_state_feedback
        predicted_probabilities=np.ones((C, P)) / C,
        state_mean=np.full((C, P), 0.3),
        bias=np.full((C, P), 0.1),
        state_var=np.full((C, P), 0.04),
        sigma_observation_noise=np.full((C, P), 0.2),
        average_state=0.25,
    )

    out_p = parent.predict_state_feedback(_dcopy(coin_state))
    # child takes same state_feedback as generated by parent
    sf = out_p["state_feedback"]
    out_c = child.predict_state_feedback(_dcopy(coin_state), state_feedback=sf)

    for k in ("state_feedback_mean", "state_feedback_var", "motor_output", "implicit"):
        assert k in out_p and k in out_c
        np.testing.assert_allclose(out_c[k], out_p[k])


# 5) sample_context parity (ensure no-new-cue path matches)
def test_sample_context_parity_existing_cue(fresh_models, seeded):
    parent, child = fresh_models
    rs = seeded(77)
    C = parent.max_contexts + 1
    P = parent.particles

    # responsibilities: normalized over contexts per particle
    R = np.zeros((C, P))
    R[0:2, :] = rs.rand(2, P)
    R /= np.sum(R, axis=0, keepdims=True)

    # Global Transition Probabilities must be correctly defined
    gqp = np.zeros((C, P))
    gqp[0:2, :] = rs.rand(2, P)
    gqp /= np.sum(gqp, axis=0, keepdims=True)

    coin_state = dict(
        cues_exist=1,
        trial=1,
        responsibilities=R,
        context=np.ones((P,), dtype=int),
        C=np.ones((P,), dtype=int),
        global_transition_probabilities=gqp,
        Q=1,  # already saw cue '1'
        global_cue_probabilities=np.zeros((child.max_cues + 2, P)),
    )

    cue = 1  # same as existing (forces "existing cue" path, not "new cue")
    parent.cues = np.array([cue], dtype=int)

    np.random.seed(123)
    out_p = parent.sample_context(_dcopy(coin_state))

    np.random.seed(123)
    out_c = child.sample_context(_dcopy(coin_state), cue=cue)

    np.testing.assert_array_equal(out_c["context"], out_p["context"])
    np.testing.assert_array_equal(out_c["C"], out_p["C"])
    np.testing.assert_allclose(
        out_c["global_transition_probabilities"], out_p["global_transition_probabilities"]
    )


# 6) update_sufficient_statistics_global_cue_probabilities parity
def test_update_stats_global_cue_probabilities_parity(fresh_models):
    parent, child = fresh_models
    P = parent.particles
    C = parent.max_contexts + 1
    Qmax = child.max_cues + 1
    cue = 3

    coin_state = dict(
        context=np.full((P,), 2, dtype=int),               # all particles in context 2
        n_cue=np.zeros((C, Qmax, P), dtype=int),
        trial=0
    )

    parent.cues = np.array([cue], dtype=int)

    out_p = parent.update_sufficient_statistics_global_cue_probabilities(_dcopy(coin_state))
    out_c = child.update_sufficient_statistics_global_cue_probabilities(_dcopy(coin_state), cue=cue)

    np.testing.assert_array_equal(out_c["n_cue"], out_p["n_cue"])


# 7) update_sufficient_statistics_for_parameters dispatch parity
def test_update_stats_for_parameters_dispatch_parity(fresh_models, monkeypatch):
    parent, child = fresh_models
    parent.infer_bias = True
    child.infer_bias = True

    calls_p = dict(gtp=0, gcp=0, dyn=0, bias=0)
    calls_c = dict(gtp=0, gcp=0, dyn=0, bias=0)

    # Parent spies
    def _gtp_p(cs): calls_p.__setitem__("gtp", calls_p["gtp"] + 1); return cs
    def _gcp_p(cs): calls_p.__setitem__("gcp", calls_p["gcp"] + 1); return cs
    def _dyn_p(cs): calls_p.__setitem__("dyn", calls_p["dyn"] + 1); return cs
    def _bias_p(cs): calls_p.__setitem__("bias", calls_p["bias"] + 1); return cs

    # Child spies (note child gcp takes a cue arg)
    def _gtp_c(cs): calls_c.__setitem__("gtp", calls_c["gtp"] + 1); return cs
    def _gcp_c(cs, cue): 
        assert cue == 2
        calls_c.__setitem__("gcp", calls_c["gcp"] + 1); 
        return cs
    def _dyn_c(cs): calls_c.__setitem__("dyn", calls_c["dyn"] + 1); return cs
    def _bias_c(cs): calls_c.__setitem__("bias", calls_c["bias"] + 1); return cs

    # Patch parent
    monkeypatch.setattr(parent, "update_sufficient_statistics_global_transition_probabilities", _gtp_p, raising=True)
    monkeypatch.setattr(parent, "update_sufficient_statistics_global_cue_probabilities", _gcp_p, raising=True)
    monkeypatch.setattr(parent, "update_sufficient_statistics_dynamics", _dyn_p, raising=True)
    monkeypatch.setattr(parent, "update_sufficient_statistics_bias", _bias_p, raising=True)

    # Patch child
    monkeypatch.setattr(child, "update_sufficient_statistics_global_transition_probabilities", _gtp_c, raising=True)
    monkeypatch.setattr(child, "update_sufficient_statistics_global_cue_probabilities", _gcp_c, raising=True)
    monkeypatch.setattr(child, "update_sufficient_statistics_dynamics", _dyn_c, raising=True)
    monkeypatch.setattr(child, "update_sufficient_statistics_bias", _bias_c, raising=True)

    # Equivalent coin_state; parent fetches cue from self.cues, child gets explicit cue
    coin_state = dict(
        cues_exist=1,
        trial=2,                       # >1 triggers dynamics path
        feedback_observed=[True, True] # trial-1 index -> True triggers bias path
    )
    parent.cues = np.array([2], dtype=int)

    parent.update_sufficient_statistics_for_parameters(_dcopy(coin_state))
    child.update_sufficient_statistics_for_parameters(_dcopy(coin_state), cue=2)

    assert calls_p == dict(gtp=1, gcp=1, dyn=1, bias=1)
    assert calls_c == dict(gtp=1, gcp=1, dyn=1, bias=1)


# --- Ensure the "matches/batch simulation" parity runs last --------------------
# If your suite defines a helper or a specific test for batch parity (e.g. 
# matches_batch_simulation() or test_rt_matches_batch_simulation_given_same_feedback),
# we trigger it here so it executes at the end of this file.

def test_zz_run_matches_batch_simulation_if_present():
    # Try a helper named `matches_batch_simulation` first
    g = globals()
    if "matches_batch_simulation" in g and callable(g["matches_batch_simulation"]):
        g["matches_batch_simulation"]()
        return

    # Otherwise, try to call an existing test if it exists in this module
    if "test_rt_matches_batch_simulation_given_same_feedback" in g and callable(g["test_rt_matches_batch_simulation_given_same_feedback"]):
        # Call the existing test function directly (re-uses its own internals/fixtures)
        g["test_rt_matches_batch_simulation_given_same_feedback"]()  # type: ignore
        return

    pytest.skip("No batch/matches simulation function found in this file.")
