# tests/test_liegroups.py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from liegroup import LieGroup, SE2, SO2, SO3, SE3, U1, UnitQuat, RAdd, RnAdd_factory, RplusMul
import functools

GROUP_CLASSES = [SE2, SO2, SO3, SE3, U1, UnitQuat, RAdd, RnAdd_factory(50), RplusMul]

N_REPEATS = 25

import traceback

def repeat(n):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(n):
                try:
                    func(*args, **kwargs)
                except AssertionError as e:
                    print("\n" + "="*60)
                    print(f"Repetition {i+1}/{n} failed for test: {func.__name__}")
                    print(f"Args: {args}")
                    print(f"Kwargs: {kwargs}")
                    print("Traceback:")
                    traceback.print_exc()
                    print("="*60 + "\n")
                    raise AssertionError(f"Failure on repetition {i+1}/{n}:\n{e}") from e
        return wrapper
    return decorator

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_identity_and_inverse(G):
    # sample a random group element
    v = np.random.randn(G.N)
    X = G.Exp(v)
    I = G.identity()

    # identity
    assert_allclose(G.compose(I, X), X, atol=1e-5)
    assert_allclose(G.compose(X, I), X, atol=1e-5)

    # inverse
    X_inv = G.inverse(X)
    assert_allclose(G.compose(X, X_inv), I, atol=1e-5)
    assert_allclose(G.compose(X_inv, X), I, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_associativity(G):
    A = G.Exp(np.random.randn(G.N))
    B = G.Exp(np.random.randn(G.N))
    C = G.Exp(np.random.randn(G.N))
    left = G.compose(A, G.compose(B, C))
    right = G.compose(G.compose(A, B), C)
    assert_allclose(left, right, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_hat_vee_roundtrip(G):
    v = np.random.randn(G.N)
    H = G.hat(v)
    assert_allclose(G.vee(H), v, atol=1e-5)
    # and the other way
    v2 = G.vee(H)
    assert_allclose(G.hat(v2), H, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_exp_log_roundtrip(G):
    # Exp ∘ Log = id on group
    v = np.random.randn(G.N)
    X = G.Exp(v)
    print(X)
    print(G.Log(X))
    assert_allclose(G.Exp(G.Log(X)), X, atol=1e-5)
    
@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_log_exp_roundtrip(G):
    # Log ∘ Exp = id on algebra
    v = np.random.randn(G.N)
    try:
        assert_allclose(G.Log(G.Exp(v)), v, atol=1e-5)
    except AssertionError:
        print(f"[WARN] Log(Exp(v)) ≠ v — fallback triggered. v = {v}")
        assert_allclose(G.Exp(G.Log(G.Exp(v))), G.Exp(v), atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_oplus_ominus_roundtrip(G):
    X = G.Exp(np.random.randn(G.N))
    v = np.random.randn(G.N)
    Y = G.oplus(X, v)
    recovered = G.ominus(X, Y)
    try:
        assert_allclose(recovered, v, atol=1e-5)
    except AssertionError:
        print(f"[WARN] ominus(oplus(X, v)) ≠ v — fallback triggered.\n v = {v}\n recovered = {recovered}")
        assert_allclose(G.Exp(recovered), G.Exp(v), atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_adjoint_conjugation(G):
    X = G.Exp(np.random.randn(G.N))
    v = np.random.randn(G.N)
    # X · Exp(v) · X⁻¹  vs  Exp(Adj_X ⋅ v)
    lhs = G.compose(G.compose(X, G.Exp(v)), G.inverse(X))
    rhs = G.Exp(G.adjoint(X) @ v)
    assert_allclose(lhs, rhs, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
def test_exp_log_near_zero_and_pi(G):
    for theta in [1e-10, np.pi - 1e-5, -np.pi + 1e-5]:
        v = np.zeros(G.N)
        v[0] = theta  # assume θ is always the first component
        X = G.Exp(v)
        X_back = G.Exp(G.Log(X))
        assert_allclose(X, X_back, atol=1e-5)
        

@pytest.mark.parametrize("G", [SO2, SO3])
@repeat(N_REPEATS)
def test_log_output_skew_symmetric(G):
    X = G.Exp(np.random.randn(G.N))
    L = G.log(X)
    assert_allclose(L + L.T, np.zeros_like(L), atol=1e-8)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_adjoint_composition(G):
    X = G.Exp(np.random.randn(G.N))
    Y = G.Exp(np.random.randn(G.N))
    Ad_X = G.adjoint(X)
    Ad_Y = G.adjoint(Y)
    Ad_XY = G.adjoint(G.compose(X, Y))
    assert_allclose(Ad_XY, Ad_X @ Ad_Y, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_inverse_involution(G):
    X = G.Exp(np.random.randn(G.N))
    assert_allclose(G.inverse(G.inverse(X)), X, atol=1e-5)
    
@pytest.mark.parametrize("G", GROUP_CLASSES)
def test_exp_zero_is_identity(G):
    zero = np.zeros(G.N)
    assert_allclose(G.Exp(zero), G.identity(), atol=1e-8)


@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_left_jacobian_inverse(G):
    v = np.random.randn(G.N)
    J = G.left_jacobian(v)
    J_inv = G.left_jacobian_inverse(v)
    I = np.eye(G.N)
    assert_allclose(J @ J_inv, I, atol=1e-5)
    assert_allclose(J_inv @ J, I, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_left_jacobian_consistency(G):
    v = np.random.randn(G.N)
    J = G.left_jacobian(v)
    delta = 1e-6 * np.random.randn(G.N)

    # Perturbation in tangent space
    v_perturbed = v + J @ delta
    X_perturbed = G.Exp(v_perturbed)
    X = G.Exp(v)

    # Theoretically close
    expected = G.compose(X, G.Exp(delta))
    assert_allclose(X_perturbed, expected, atol=1e-5)
    
@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_group_closure(G):
    X = G.Exp(np.random.randn(G.N))
    Y = G.Exp(np.random.randn(G.N))
    Z = G.compose(X, Y)
    assert np.isfinite(Z).all()  # No NaNs or infs
    

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_lie_bracket_antisymmetry(G):
    v = np.random.randn(G.N)
    w = np.random.randn(G.N)
    bracket = G.lie_bracket(v, w)
    bracket_flip = G.lie_bracket(w, v)
    assert_allclose(bracket + bracket_flip, np.zeros_like(bracket), atol=1e-8)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_lie_bracket_jacobi_identity(G):
    v = np.random.randn(G.N)
    w = np.random.randn(G.N)
    z = np.random.randn(G.N)
    term1 = G.lie_bracket(v, G.lie_bracket(w, z))
    term2 = G.lie_bracket(w, G.lie_bracket(z, v))
    term3 = G.lie_bracket(z, G.lie_bracket(v, w))
    total = term1 + term2 + term3
    assert_allclose(total, np.zeros_like(total), atol=1e-6)


@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_lie_bracket_self_zero(G):
    v = np.random.randn(G.N)
    assert_allclose(G.lie_bracket(v, v), np.zeros(G.N), atol=1e-8)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_log_inverse_is_minus_adjoint(G):
    X = G.Exp(np.random.randn(G.N))
    v_log = G.Log(X)
    v_inv_log = G.Log(G.inverse(X))
    lhs = v_inv_log
    rhs = -G.adjoint(X) @ v_log
    assert_allclose(lhs, rhs, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_adjoint_preserves_bracket(G):
    X = G.Exp(np.random.randn(G.N))
    v = np.random.randn(G.N)
    w = np.random.randn(G.N)
    lhs = G.adjoint(X) @ G.lie_bracket(v, w)
    rhs = G.lie_bracket(G.adjoint(X) @ v, G.adjoint(X) @ w)
    assert_allclose(lhs, rhs, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_distance_properties(G):
    X = G.Exp(np.random.randn(G.N))
    Y = G.Exp(np.random.randn(G.N))
    d1 = G.distance(X, Y)
    d2 = G.distance(Y, X)

    # Distance is non-negative
    assert d1 >= 0
    assert d2 >= 0

    # Distance is zero if and only if same element
    assert_allclose(G.distance(X, X), 0.0, atol=1e-8)

    # Symmetry
    assert_allclose(d1, d2, atol=1e-5)

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_random_normalized(G):
    X = G.randu()
    X_normed = G.normalize(X)
    assert np.isfinite(X_normed).all()

@pytest.mark.parametrize("G", GROUP_CLASSES)
@repeat(N_REPEATS)
def test_distance_identity(G):
    X = G.randu()
    X = G.normalize(X)
    assert_allclose(G.distance(X, X), 0.0, atol=1e-8)

@pytest.mark.parametrize("G", GROUP_CLASSES)
def test_randn_zero_is_identity(G):
    X = G.randn_G(scale=0.0)
    I = G.identity()
    assert_allclose(X, I, atol=1e-8)
