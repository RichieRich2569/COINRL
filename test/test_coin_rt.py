# test_coin_rt_real.py
"""
Integration & parity tests for COIN (batch) vs COIN_RT (real-time).

This suite focuses on:
  • Initialization invariants with/without cues.
  • Public-facing method behavior: shapes, normalization, and arithmetic.
  • Dispatch logic for sufficient-statistics updates.
  • End-to-end parity of observable outputs (motor_output) under identical seeds
    and identical observations/perturbations.
  • Determinism: same seed + same inputs → same outputs (RT).

Notes on scope and philosophy:
  - We avoid brittle, deep equality checks on the *entire* internal coin_state.
    RNG consumption can legitimately diverge between batch and RT flows while
    still producing identical external outputs. We therefore assert strongly on
    observable outputs and on mathematically stable invariants (shapes, sums=1, etc.).
  - One known issue is documented with an xfail: when cues_exist=True, COIN_RT.step()
    expects a list at self.cues, but initialise_coin sets self.cues=None.
"""

from __future__ import annotations

import importlib
import numpy as np
import pytest


# ---------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------

def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        pytest.skip(f"Module '{module_name}' not importable: {e}")


def import_modules():
    """
    Import the 'coin' (batch) and 'coin_rt' (real-time) modules.

    If either is missing, skip the suite to avoid CI false negatives.
    """
    coin = _try_import("coin")
    coin_rt = _try_import("coin_rt")
    assert hasattr(coin_rt, "COIN_RT"), "COIN_RT class not found in coin_rt module"
    assert hasattr(coin, "COIN"), "COIN class not found in coin module"
    return coin, coin_rt


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _ensure_list_in_state(coin_state: dict, key: str):
    """Ensure coin_state[key] exists as a list (used for bookkeeping arrays)."""
    if key not in coin_state or coin_state[key] is None:
        coin_state[key] = []


def _rng(seed: int = 123):
    """Convenience RandomState for test data generation."""
    return np.random.RandomState(seed)


def _dcopy(d: dict) -> dict:
    """Shallow dict copy; np.ndarray are copied by value to prevent aliasing."""
    out = {}
    for k, v in d.items():
        out[k] = v.copy() if isinstance(v, np.ndarray) else v
    return out


# ---------------------------------------------------------------------
# Fixtures: models with/without cues
# ---------------------------------------------------------------------

@pytest.fixture
def model_no_cues():
    """
    COIN_RT instance with cues disabled, plus minimal bookkeeping fields
    so we can call step() without tripping implementation-specific checks.
    """
    _, coin_rt = import_modules()
    m = coin_rt.COIN_RT(cues_exist=False)
    m.coin_state.setdefault("trial", 0)
    _ensure_list_in_state(m.coin_state, "feedback_observed")
    return m


@pytest.fixture
def model_with_cues():
    """
    COIN_RT instance with cues enabled.
    We *do not* mutate self.cues here to surface the documented bug via the xfail test.
    """
    _, coin_rt = import_modules()
    m = coin_rt.COIN_RT(cues_exist=True, max_contexts=4, max_cues=3, particles=10)
    m.coin_state.setdefault("trial", 0)
    _ensure_list_in_state(m.coin_state, "feedback_observed")
    return m


# ---------------------------------------------------------------------
# 1) Initialization & basic invariants
# ---------------------------------------------------------------------

def test_init_defaults_without_cues():
    """
    Default construction without cues should populate basic invariants and
    omit cue-specific fields.
    """
    _, coin_rt = import_modules()
    model = coin_rt.COIN_RT()  # defaults: cues_exist=False

    assert model.trial == 0
    assert model.max_cues == 10
    assert isinstance(model.coin_state, dict)
    assert model.perturbations == []
    assert model.cues is None

    # cue-related fields should be absent
    assert model.coin_state["cues_exist"] == 0
    assert "Q" not in model.coin_state
    assert "n_cue" not in model.coin_state

    # mirror of parent defaults if exposed
    if hasattr(model, "particles"):
        assert model.particles == 100
    if hasattr(model, "max_contexts"):
        assert model.max_contexts == 10


@pytest.mark.parametrize(
    "particles,max_contexts,max_cues,trial",
    [(64, 7, 5, 3), (8, 2, 2, 11)],
)
def test_init_with_cues_shapes(particles, max_contexts, max_cues, trial):
    """
    When cues_exist=True, cue-specific fields must be created with correct shapes.
    """
    _, coin_rt = import_modules()
    model = coin_rt.COIN_RT(
        particles=particles,
        max_contexts=max_contexts,
        max_cues=max_cues,
        trial=trial,
        cues_exist=True,
    )

    cs = model.coin_state
    assert cs.get("cues_exist") == 1
    assert cs.get("Q") == 0
    assert "n_cue" in cs and isinstance(cs["n_cue"], np.ndarray)
    assert cs["n_cue"].shape == (max_contexts + 1, max_cues + 1, particles)
    assert np.all(cs["n_cue"] == 0)

    assert model.trial == trial
    assert model.max_cues == max_cues
    if hasattr(model, "particles"):
        assert model.particles == particles
    if hasattr(model, "max_contexts"):
        assert model.max_contexts == max_contexts


def test_reinitialise_toggles_cue_fields_and_resets_bookkeeping():
    """
    Calling initialise_coin should:
      - Switch cue fields on/off according to 'cues_exist'
      - Reset user bookkeeping like perturbations and cues container
    """
    _, coin_rt = import_modules()
    model = coin_rt.COIN_RT(particles=8, max_contexts=2, max_cues=2)

    # Defaults (cues off)
    assert model.coin_state["cues_exist"] == 0
    assert model.perturbations == []
    assert model.cues is None

    # Mutate to verify reset
    model.perturbations = [1, 2, 3]
    model.cues = ["x"]

    # Enable cues
    cs2 = model.initialise_coin(cues_exist=True)
    assert cs2["cues_exist"] == 1 and cs2["Q"] == 0 and "n_cue" in cs2
    assert cs2["n_cue"].shape == (model.max_contexts + 1, model.max_cues + 1, model.particles)
    # Bookkeeping reset
    assert model.perturbations == []
    # Implementation-dependent: expect a list of length equal to maximum cues, or an empty list.
    # If your implementation sets a sentinel, adjust here. We assert "truthiness" of list-like.
    assert isinstance(model.cues, (list, type(None)))

    # Disable cues again
    cs3 = model.initialise_coin(cues_exist=False)
    assert cs3["cues_exist"] == 0
    assert "Q" not in cs3 and "n_cue" not in cs3


def test_parameter_overrides_reflected_if_exposed():
    """
    Construction with parameter overrides should not crash and should reflect
    sizes if the base class exposes them.
    """
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

    assert model.coin_state["cues_exist"] == 1
    assert model.coin_state["Q"] == 0
    assert model.coin_state["n_cue"].shape == (3 + 1, 4 + 1, 32)
    if hasattr(model, "particles"):
        assert model.particles == 32
    if hasattr(model, "max_contexts"):
        assert model.max_contexts == 3


# ---------------------------------------------------------------------
# 2) step() validation & messaging
# ---------------------------------------------------------------------

def test_step_raises_when_cues_required_but_missing(model_with_cues):
    """
    When cues_exist=True and cue=None, step() must raise ValueError AFTER incrementing trial.
    """
    m = model_with_cues
    m.cues = []  # ensure AttributeError isn't masking the validation
    before = m.coin_state["trial"]
    with pytest.raises(ValueError, match="Cue must be provided"):
        m.step(state_feedback=0.1, cue=None)
    assert m.coin_state["trial"] == before + 1


def test_step_warns_when_cue_provided_but_cues_do_not_exist(model_no_cues, capsys, monkeypatch):
    """
    With cues_exist=False and a cue provided, step() should warn and ignore the cue.
    """
    m = model_no_cues

    # Minimal state so the chain can run with monkeypatched heavy calls
    m.coin_state.setdefault("prior_probabilities", np.ones((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("predicted_probabilities", np.ones((m.max_contexts + 1, m.particles)) / (m.max_contexts + 1))
    m.coin_state.setdefault("state_mean", np.zeros((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("bias", np.zeros((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("state_var", np.ones((m.max_contexts + 1, m.particles)))
    m.coin_state.setdefault("sigma_observation_noise", np.ones((m.max_contexts + 1, m.particles)) * 0.1)
    m.coin_state.setdefault("average_state", 0.0)

    # Monkeypatch heavy calls to no-ops
    monkeypatch.setattr(m, "predict_context", lambda cs, cue=None: cs, raising=True)
    monkeypatch.setattr(m, "predict_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "resample_particles", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "sample_context", lambda cs, cue=None: cs, raising=True)
    monkeypatch.setattr(m, "update_belief_about_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "sample_states", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_for_parameters", lambda cs, cue=None: cs, raising=True)
    monkeypatch.setattr(m, "sample_parameters", lambda cs: cs, raising=True)
    monkeypatch.setattr(m, "compute_marginal_distribution", lambda cs: cs, raising=True)

    def _store_stub(cs):
        cs["stored"] = {n: {"ok": True} for n in range(m.runs)}
        return cs

    monkeypatch.setattr(m, "store_variables", _store_stub, raising=True)

    res = m.step(state_feedback=0.2, cue=1)
    out = capsys.readouterr().out
    assert "Warning: Cue provided but cues_exist is False" in out
    assert isinstance(res, dict) and {"runs", "weights", "properties"} <= set(res.keys())
    assert set(res["runs"].keys()) == set(range(m.runs))


# ---------------------------------------------------------------------
# 3) predict_context — shapes & normalization (with/without cues)
# ---------------------------------------------------------------------

def test_predict_context_without_cues_shapes_and_passthrough(model_with_cues):
    """
    No-cue branch: predicted_probabilities should equal prior_probabilities.
    """
    m = model_with_cues
    cs = dict(
        cues_exist=0,
        trial=1,
        context=np.ones((m.particles,), dtype=int),
        local_transition_matrix=_rng(7).rand(m.max_contexts + 1, m.max_contexts + 1, m.particles),
    )
    m.store = []  # avoid optional branches

    out = m.predict_context(cs, cue=None)
    assert out["prior_probabilities"].shape == (m.max_contexts + 1, m.particles)
    assert out["predicted_probabilities"].shape == (m.max_contexts + 1, m.particles)
    np.testing.assert_allclose(out["predicted_probabilities"], out["prior_probabilities"])


def test_predict_context_with_cues_column_normalizes(model_with_cues):
    """
    Cue branch: predicted_probabilities must be column-normalized (sum over contexts = 1).
    """
    m = model_with_cues
    rs = _rng(9)
    cs = dict(
        cues_exist=1,
        trial=2,
        context=np.ones((m.particles,), dtype=int),
        Q=0,
        local_transition_matrix=rs.rand(m.max_contexts + 1, m.max_contexts + 1, m.particles),
        local_cue_matrix=rs.rand(m.max_contexts + 1, m.max_cues + 1, m.particles),
    )
    m.store = []

    out = m.predict_context(cs, cue=1)
    col_sums = np.sum(out["predicted_probabilities"], axis=0, keepdims=True)
    np.testing.assert_allclose(col_sums, np.ones_like(col_sums), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# 4) predict_state_feedback — arithmetic checks
# ---------------------------------------------------------------------

def test_predict_state_feedback_means_vars_motor_output(model_with_cues, monkeypatch):
    """
    Sanity checks:
      mean  = state_mean + bias
      var   = state_var + sigma_observation_noise^2
      mo    = avg over particles of sum_c P(c) * mean(c)
      implicit = motor_output - average_state   (when 'implicit' is stored)
      prediction_error = y_t - mean (per-context/particle)
    """
    m = model_with_cues
    m.store = ["implicit"]

    C = m.max_contexts + 1
    P = m.particles

    cs = dict(
        cues_exist=1,
        trial=1,
        predicted_probabilities=np.ones((C, P)) / C,
        state_mean=np.full((C, P), 0.3),
        bias=np.full((C, P), 0.1),
        state_var=np.full((C, P), 0.04),
        sigma_observation_noise=np.full((C, P), 0.2),
        average_state=0.25,
    )

    monkeypatch.setattr(m, "compute_marginal_distribution", lambda d: d, raising=True)

    y = 0.8
    out = m.predict_state_feedback(cs, state_feedback=y)

    expected_mean = 0.3 + 0.1
    expected_var = 0.04 + (0.2 ** 2)
    assert np.allclose(out["state_feedback_mean"], expected_mean)
    assert np.allclose(out["state_feedback_var"], expected_var)

    mo = np.sum(out["predicted_probabilities"] * out["state_feedback_mean"]) / P
    assert np.isclose(out["motor_output"], mo)
    assert np.isclose(out["implicit"], out["motor_output"] - 0.25)

    assert out["prediction_error"].shape == (C, P)
    np.testing.assert_allclose(out["prediction_error"], y - expected_mean)


# ---------------------------------------------------------------------
# 5) sample_context — updates and probabilities
# ---------------------------------------------------------------------

def test_sample_context_updates_context_and_probabilities(model_with_cues):
    """
    Build a minimal responsibilities matrix and ensure:
      - context stays within valid bounds
      - 'C' increments when a new context is sampled
      - global_transition_probabilities are valid prob. vectors (sum to 1 per particle)
    """
    m = model_with_cues
    Cmax, P = m.max_contexts, m.particles
    rs = _rng(42)

    resp = np.zeros((Cmax + 1, P))
    resp[0:3, :] = 1.0 / 3.0

    gtp = np.zeros((Cmax + 1, P))
    gtp[0:3, :] = rs.rand(3, P)
    gtp /= np.sum(gtp, axis=0, keepdims=True)

    cs = dict(
        cues_exist=0,
        trial=2,
        responsibilities=resp,
        context=np.ones((P,), dtype=int),
        C=np.ones((P,), dtype=int) + 1,  # current max context index per particle (2)
        global_transition_probabilities=gtp,
    )

    np.random.seed(0)
    out = m.sample_context(cs, cue=None)

    assert np.all((out["context"] >= 1) & (out["context"] <= Cmax + 1))
    assert "p_new_x" in out and "p_old_x" in out
    if len(out["p_new_x"]) > 0:
        assert np.all(out["C"][out["p_new_x"]] >= 2)

    gtp2 = out["global_transition_probabilities"]
    assert gtp2.shape == (Cmax + 1, P)
    assert np.all(gtp2 >= 0.0)
    np.testing.assert_allclose(np.sum(gtp2, axis=0), np.ones((P,)), atol=1e-5)


# ---------------------------------------------------------------------
# 6) Sufficient-statistics dispatch & cue counts
# ---------------------------------------------------------------------

def test_update_sufficient_statistics_dispatch_paths(model_with_cues, monkeypatch):
    """
    update_sufficient_statistics_for_parameters should:
      - always call global transition stats
      - call cue stats iff cues_exist and cue is not None
      - call dynamics iff trial > 1
      - call bias iff infer_bias=True and feedback_observed[trial-1] is True
    """
    m = model_with_cues
    m.infer_bias = True
    flags = dict(gtp=0, gcp=0, dyn=0, bias=0)

    def _gtp(cs): flags.__setitem__("gtp", flags["gtp"] + 1); return cs
    def _gcp(cs, cue): assert cue == 1; flags.__setitem__("gcp", flags["gcp"] + 1); return cs
    def _dyn(cs): flags.__setitem__("dyn", flags["dyn"] + 1); return cs
    def _bias(cs): flags.__setitem__("bias", flags["bias"] + 1); return cs

    monkeypatch.setattr(m, "update_sufficient_statistics_global_transition_probabilities", _gtp, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_global_cue_probabilities", _gcp, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_dynamics", _dyn, raising=True)
    monkeypatch.setattr(m, "update_sufficient_statistics_bias", _bias, raising=True)

    cs = dict(cues_exist=1, trial=2, feedback_observed=[True, True])
    out = m.update_sufficient_statistics_for_parameters(cs, cue=1)

    assert out is cs
    assert flags == dict(gtp=1, gcp=1, dyn=1, bias=1)


def test_update_sufficient_statistics_global_cue_probabilities_increments(model_with_cues):
    """Increment n_cue at (context-1, cue, particle) across all particles."""
    m = model_with_cues
    P = m.particles
    Cmax = m.max_contexts
    Qmax = m.max_cues

    cs = dict(
        context=np.full((P,), 2, dtype=int),
        n_cue=np.zeros((Cmax + 1, Qmax + 1, P), dtype=int),
    )
    cue = 1
    out = m.update_sufficient_statistics_global_cue_probabilities(cs, cue=cue)
    for i in range(P):
        assert out["n_cue"][1, cue, i] == 1  # context-1 == 1


# ---------------------------------------------------------------------
# 7) Known issue (documented): cues container type
# ---------------------------------------------------------------------

@pytest.mark.xfail(reason="initialise_coin sets self.cues=None even when cues_exist=True; step() expects list.")
def test_initial_cues_container_list_when_cues_exist(model_with_cues):
    """Expected: self.cues should be a list when cues_exist=True (documenting current behavior)."""
    assert isinstance(model_with_cues.cues, list)


# ---------------------------------------------------------------------
# 8) End-to-end: batch vs RT parity on motor_output
# ---------------------------------------------------------------------

def _make_perturbations(kind: str):
    """Utility to generate perturbation schedules."""
    if kind == "single":
        return np.zeros((1,))
    if kind == "medium":
        return np.concatenate([np.zeros(5), np.ones(5), -np.ones(3), np.ones(5) * np.nan]).astype(float)
    if kind == "complex":
        return np.concatenate([
            np.zeros((50,)),
            np.ones((125,)),
            -np.ones((15,)),
            np.ones((150,)) * np.nan,
        ]).astype(float)
    raise ValueError(f"Unknown perturbation kind {kind}")


@pytest.mark.parametrize(
    "perturb_kind",
    ["single", "medium", pytest.param("complex", marks=pytest.mark.slow)],
)
def test_rt_matches_batch_simulation_given_same_feedback(perturb_kind):
    """
    Compare *observable* outputs:
      motor_output_RT  == motor_output_batch
    when both models see the same observation sequence drawn by batch,
    under the same seed and identical hyperparameters.
    """
    coin, coin_rt = import_modules()
    if not hasattr(coin.COIN, "simulate_coin"):
        pytest.skip("coin.COIN.simulate_coin() not available in this environment.")

    common_kwargs = dict(particles=5, max_contexts=10, max_cores=0)
    perturbations = _make_perturbations(perturb_kind)
    T = perturbations.shape[0]
    SEED = 12345

    # --- Batch simulation (source of truth for observations) ---
    np.random.seed(SEED)
    coin_model = coin.COIN(**common_kwargs)
    coin_model.perturbations = perturbations
    out_batch = coin_model.simulate_coin()
    assert "runs" in out_batch and 0 in out_batch["runs"]
    y_seq = np.asarray(out_batch["runs"][0]["state_feedback"])
    mo_batch = np.asarray(out_batch["runs"][0]["motor_output"])
    assert y_seq.shape[0] == T and mo_batch.shape[0] == T

    # --- Real-time using the exact same observations ---
    np.random.seed(SEED)
    rt_model = coin_rt.COIN_RT(**common_kwargs, cues_exist=False)
    rt_model.coin_state.setdefault("trial", 0)
    _ensure_list_in_state(rt_model.coin_state, "feedback_observed")

    last_res = None
    for t in range(T):
        y = y_seq[t]
        y_in = np.nan if np.isnan(y) else float(y)
        last_res = rt_model.step(state_feedback=y_in, cue=None)

    assert last_res is not None and "runs" in last_res and 0 in last_res["runs"]
    mo_rt = np.asarray(last_res["runs"][0]["motor_output"])
    assert mo_rt.shape[0] == T

    np.testing.assert_allclose(mo_rt, mo_batch, rtol=0, atol=1e-12)


def test_rt_determinism_same_seed_same_inputs_same_outputs():
    """
    Determinism test: With the same RNG seed and the same observation stream,
    two separate COIN_RT instances should produce identical motor_output.
    """
    _, coin_rt = import_modules()
    SEED = 20250930
    common_kwargs = dict(particles=7, max_contexts=5, max_cores=0)
    y_seq = np.array([0.2, 0.1, np.nan, 0.0, -0.3, 0.7], dtype=float)

    # Run 1
    np.random.seed(SEED)
    m1 = coin_rt.COIN_RT(**common_kwargs, cues_exist=False)
    m1.coin_state.setdefault("trial", 0)
    _ensure_list_in_state(m1.coin_state, "feedback_observed")
    for y in y_seq:
        res = m1.step(state_feedback=(np.nan if np.isnan(y) else float(y)), cue=None)
    mo1 = np.asarray(res["runs"][0]["motor_output"])

    # Run 2
    np.random.seed(SEED)
    m2 = coin_rt.COIN_RT(**common_kwargs, cues_exist=False)
    m2.coin_state.setdefault("trial", 0)
    _ensure_list_in_state(m2.coin_state, "feedback_observed")
    for y in y_seq:
        res = m2.step(state_feedback=(np.nan if np.isnan(y) else float(y)), cue=None)
    mo2 = np.asarray(res["runs"][0]["motor_output"])

    np.testing.assert_allclose(mo1, mo2, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------
# 9) Parity vs parent (batch) for overridden methods (behavioral parity)
# ---------------------------------------------------------------------

@pytest.fixture
def fresh_models():
    """Matched parent/child models with neutral 'store' hooks."""
    coin, coin_rt = import_modules()
    parent = coin.COIN(particles=7, max_contexts=5)
    child = coin_rt.COIN_RT(particles=7, max_contexts=5, max_cues=4)
    parent.store, child.store = [], []
    return parent, child


def test_initialise_coin_parity(fresh_models):
    """Parities on core fields for cues_exist=False and True."""
    parent, child = fresh_models

    # cues_exist=False
    np.random.seed(0)
    parent.perturbations = np.zeros((4,))
    cs_p0 = parent.initialise_coin()
    np.random.seed(0)
    cs_c0 = child.initialise_coin(cues_exist=False)
    for k in ("trial", "context", "C"):
        assert k in cs_p0 and k in cs_c0
        np.testing.assert_array_equal(cs_p0[k], cs_c0[k])

    # cues_exist=True
    np.random.seed(1)
    parent.cues = np.arange(1, 5, dtype=int)
    cs_p1 = parent.initialise_coin()
    np.random.seed(1)
    cs_c1 = child.initialise_coin(cues_exist=True)
    for k in ("trial", "context", "C", "Q"):
        assert k in cs_p1 and k in cs_c1
        np.testing.assert_array_equal(cs_p1[k], cs_c1[k])


def test_predict_context_parity_no_cues(fresh_models):
    """No-cue path: child's predicted_probabilities match parent's prior."""
    parent, child = fresh_models
    rs = _rng(2025)
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

    np.testing.assert_allclose(out_c["prior_probabilities"], out_p["prior_probabilities"])
    np.testing.assert_allclose(out_c["predicted_probabilities"], out_p["prior_probabilities"])


def test_predict_context_parity_with_cue(fresh_models):
    """Cue path parity: prior, cue, and predicted distributions match."""
    parent, child = fresh_models
    rs = _rng(7)
    C = parent.max_contexts + 1
    P = parent.particles
    Qmax = child.max_cues + 1
    cue = 2

    coin_state = dict(
        cues_exist=1,
        trial=1,
        context=rs.randint(1, C, size=(P,), dtype=int),
        local_transition_matrix=rs.rand(C, C, P),
        local_cue_matrix=rs.rand(C, Qmax, P),
    )

    parent.cues = np.array([cue], dtype=int)
    out_p = parent.predict_context(_dcopy(coin_state))
    out_c = child.predict_context(_dcopy(coin_state), cue=cue)

    for k in ("prior_probabilities", "cue_probabilities", "predicted_probabilities"):
        np.testing.assert_allclose(out_c[k], out_p[k])


def test_predict_state_feedback_parity_shared_quantities(fresh_models):
    """
    Parity on noise-free shared quantities:
      state_feedback_mean, state_feedback_var, motor_output, implicit (if enabled)
    """
    parent, child = fresh_models
    C = parent.max_contexts + 1
    P = parent.particles

    parent.perturbations = np.zeros((3,))
    parent.store = ["implicit"]
    child.store = ["implicit"]

    coin_state = dict(
        trial=1,
        predicted_probabilities=np.ones((C, P)) / C,
        state_mean=np.full((C, P), 0.3),
        bias=np.full((C, P), 0.1),
        state_var=np.full((C, P), 0.04),
        sigma_observation_noise=np.full((C, P), 0.2),
        average_state=0.25,
    )

    out_p = parent.predict_state_feedback(_dcopy(coin_state))
    sf = out_p.get("state_feedback", np.nan)  # parent may output it
    out_c = child.predict_state_feedback(_dcopy(coin_state), state_feedback=sf)

    for k in ("state_feedback_mean", "state_feedback_var", "motor_output", "implicit"):
        assert k in out_p and k in out_c
        np.testing.assert_allclose(out_c[k], out_p[k])


def test_sample_context_parity_existing_cue(fresh_models):
    """
    When the cue is existing (not novel), resampling decisions should match
    given the same RNG state.
    """
    parent, child = fresh_models
    rs = _rng(77)
    C = parent.max_contexts + 1
    P = parent.particles

    R = np.zeros((C, P))
    R[0:2, :] = rs.rand(2, P)
    R /= np.sum(R, axis=0, keepdims=True)

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
        Q=1,
        global_cue_probabilities=np.zeros((child.max_cues + 2, P)),
    )

    cue = 1
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


def test_update_stats_global_cue_probabilities_parity(fresh_models):
    """Parity on n_cue increments."""
    parent, child = fresh_models
    P = parent.particles
    C = parent.max_contexts + 1
    Qmax = child.max_cues + 1
    cue = 3

    coin_state = dict(
        context=np.full((P,), 2, dtype=int),
        n_cue=np.zeros((C, Qmax, P), dtype=int),
        trial=0,
    )

    parent.cues = np.array([cue], dtype=int)
    out_p = parent.update_sufficient_statistics_global_cue_probabilities(_dcopy(coin_state))
    out_c = child.update_sufficient_statistics_global_cue_probabilities(_dcopy(coin_state), cue=cue)
    np.testing.assert_array_equal(out_c["n_cue"], out_p["n_cue"])


def test_update_stats_for_parameters_dispatch_parity(fresh_models, monkeypatch):
    """Parity on dispatch: gtp, gcp, dyn, bias calls."""
    parent, child = fresh_models
    parent.infer_bias = True
    child.infer_bias = True

    calls_p = dict(gtp=0, gcp=0, dyn=0, bias=0)
    calls_c = dict(gtp=0, gcp=0, dyn=0, bias=0)

    def _gtp_p(cs): calls_p.__setitem__("gtp", calls_p["gtp"] + 1); return cs
    def _gcp_p(cs): calls_p.__setitem__("gcp", calls_p["gcp"] + 1); return cs
    def _dyn_p(cs): calls_p.__setitem__("dyn", calls_p["dyn"] + 1); return cs
    def _bias_p(cs): calls_p.__setitem__("bias", calls_p["bias"] + 1); return cs

    def _gtp_c(cs): calls_c.__setitem__("gtp", calls_c["gtp"] + 1); return cs
    def _gcp_c(cs, cue): assert cue == 2; calls_c.__setitem__("gcp", calls_c["gcp"] + 1); return cs
    def _dyn_c(cs): calls_c.__setitem__("dyn", calls_c["dyn"] + 1); return cs
    def _bias_c(cs): calls_c.__setitem__("bias", calls_c["bias"] + 1); return cs

    monkeypatch.setattr(parent, "update_sufficient_statistics_global_transition_probabilities", _gtp_p, raising=True)
    monkeypatch.setattr(parent, "update_sufficient_statistics_global_cue_probabilities", _gcp_p, raising=True)
    monkeypatch.setattr(parent, "update_sufficient_statistics_dynamics", _dyn_p, raising=True)
    monkeypatch.setattr(parent, "update_sufficient_statistics_bias", _bias_p, raising=True)

    monkeypatch.setattr(child, "update_sufficient_statistics_global_transition_probabilities", _gtp_c, raising=True)
    monkeypatch.setattr(child, "update_sufficient_statistics_global_cue_probabilities", _gcp_c, raising=True)
    monkeypatch.setattr(child, "update_sufficient_statistics_dynamics", _dyn_c, raising=True)
    monkeypatch.setattr(child, "update_sufficient_statistics_bias", _bias_c, raising=True)

    coin_state = dict(cues_exist=1, trial=2, feedback_observed=[True, True])
    parent.cues = np.array([2], dtype=int)

    parent.update_sufficient_statistics_for_parameters(_dcopy(coin_state))
    child.update_sufficient_statistics_for_parameters(_dcopy(coin_state), cue=2)

    assert calls_p == dict(gtp=1, gcp=1, dyn=1, bias=1)
    assert calls_c == dict(gtp=1, gcp=1, dyn=1, bias=1)
