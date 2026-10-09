import numpy as np
import pytest

from swallowtail_physics import find_equilibrium, intersect


# --------------------------------------------------------------------------- #
# Curve generators
# --------------------------------------------------------------------------- #

def polynomial_swallowtail(n_half=1001, s_max=2.5):
    """Swallowtail from P(s) = 3s - s^3, G(s) = 3s^2/2 - 3s^4/4.

    dG/dP = s plays the role of a volume, and the cusps sit at s = +-1. The
    exact curve crosses itself only at s = +-sqrt(3), i.e. at (P, G) = (0, -9/4).
    The grid in s is mirror-symmetric, so the sampled polyline is exactly
    symmetric under P -> -P.
    """
    s_pos = np.linspace(0.0, s_max, n_half)
    s = np.concatenate((-s_pos[:0:-1], s_pos))
    return 3 * s - s**3, 1.5 * s**2 - 0.75 * s**4, s


def vdw_isotherm(T, v):
    """Reduced van der Waals pressure and Gibbs free energy (up to f(T))."""
    P = 8 * T / (3 * v - 1) - 3 / v**2
    G = -(8 * T / 3) * np.log(3 * v - 1) - 3 / v + P * v
    return P, G


def maxwell_coexistence(T, P_lo, P_hi):
    """Reference coexistence point by bisection on G_gas(P) - G_liquid(P).

    Independent of the polyline algorithm: at each trial pressure the liquid
    and gas volumes are the smallest and largest roots of the vdW cubic
    3P v^3 - (P + 8T) v^2 + 9v - 3 = 0.
    """
    def branch_volumes(P):
        v = np.sort(np.roots([3 * P, -(P + 8 * T), 9, -3]).real)
        return v[0], v[-1]

    def delta_G(P):
        v_l, v_g = branch_volumes(P)
        return vdw_isotherm(T, v_g)[1] - vdw_isotherm(T, v_l)[1]

    assert delta_G(P_lo) < 0 < delta_G(P_hi)
    for _ in range(200):
        P_mid = 0.5 * (P_lo + P_hi)
        if delta_G(P_mid) < 0:
            P_lo = P_mid
        else:
            P_hi = P_mid
    P_eq = 0.5 * (P_lo + P_hi)
    v_l, _ = branch_volumes(P_eq)
    return P_eq, vdw_isotherm(T, v_l)[1]


# Minimal hand-built swallowtail: branch G = P from (0, 0), cusp at (5, 5.5),
# back-bending branch to the second cusp at (-1, 3.5), then the branch
# G = 3 - P/2 out to (6, 0). The branches cross exactly at (2, 2).
LOOP_P = np.array([0.0, 4.0, 5.0, -1.0, 6.0])
LOOP_G = np.array([0.0, 4.0, 5.5, 3.5, 0.0])


# --------------------------------------------------------------------------- #
# Swallowtail crossings
# --------------------------------------------------------------------------- #

def test_programmatic_swallowtail_matches_polyline_crossing_exactly():
    P, G, s = polynomial_swallowtail()

    P_eq, G_eq = find_equilibrium(P, G)

    # The polyline is symmetric under P -> -P, so its single crossing lies on
    # P = 0. There it sits on the first branch (s < -1), which is monotonic in
    # P, so the expected G is that branch's linear interpolant at P = 0.
    first = s < -1
    G_expected = np.interp(0.0, P[first][::-1], G[first][::-1])
    assert P_eq == pytest.approx(0.0, abs=1e-12)
    assert G_eq == pytest.approx(G_expected, abs=1e-12)


def test_programmatic_swallowtail_converges_to_analytic_crossing():
    P, G, s = polynomial_swallowtail(n_half=1001)

    P_eq, G_eq = find_equilibrium(P, G)

    # Linear interpolation error near the crossing: |G''(P)| h_P^2 / 8 with
    # G'' = 1 / P'(s) = -1/6 and h_P = |P'(s)| h_s = 6 h_s at s = -sqrt(3).
    h_s = s[1] - s[0]
    bound = (6 * h_s) ** 2 / (8 * 6)
    assert P_eq == pytest.approx(0.0, abs=1e-12)
    assert abs(G_eq - (-9 / 4)) <= bound


def test_hand_built_loop_gives_exact_rational_crossing():
    assert find_equilibrium(LOOP_P, LOOP_G) == pytest.approx((2.0, 2.0), rel=1e-12)


def test_result_does_not_depend_on_traversal_direction():
    forward = find_equilibrium(LOOP_P, LOOP_G)
    backward = find_equilibrium(LOOP_P[::-1], LOOP_G[::-1])
    assert backward == pytest.approx(forward, rel=1e-12)


@pytest.mark.parametrize("p_scale, g_scale", [(1e5, 1e-3), (1e-6, 1e8), (-2.0, 3.0)])
def test_result_is_independent_of_units(p_scale, g_scale):
    # Offsets are applied before scaling: an offset far larger than the loop
    # itself would discard significant digits of the input data.
    P_eq, G_eq = find_equilibrium(p_scale * (LOOP_P + 7.0), g_scale * (LOOP_G - 1.0))
    assert P_eq == pytest.approx(p_scale * 9.0, rel=1e-12)
    assert G_eq == pytest.approx(g_scale * 1.0, rel=1e-12)


def test_van_der_waals_isotherm_matches_maxwell_construction():
    T = 0.9
    v = np.geomspace(0.45, 10.0, 5000)
    P, G = vdw_isotherm(T, v)
    P_ref, G_ref = maxwell_coexistence(T, P_lo=0.5, P_hi=0.7)
    assert P_ref == pytest.approx(0.6470, abs=1e-4)  # tabulated vdW value

    P_eq, G_eq = find_equilibrium(P, G)

    assert P_eq == pytest.approx(P_ref, rel=1e-6)
    assert G_eq == pytest.approx(G_ref, rel=1e-6)


def test_lists_are_accepted_and_floats_returned():
    P_eq, G_eq = find_equilibrium(LOOP_P.tolist(), LOOP_G.tolist())
    assert type(P_eq) is float and type(G_eq) is float


def test_first_crossing_along_the_curve_is_returned():
    # Two loops in sequence: crossings at (2, 2) and, shifted by (10, -10), at (12, -8).
    P = np.concatenate((LOOP_P, LOOP_P + 10.0))
    G = np.concatenate((LOOP_G, LOOP_G - 10.0))

    assert find_equilibrium(P, G) == pytest.approx((2.0, 2.0), rel=1e-12)
    assert find_equilibrium(P[::-1], G[::-1]) == pytest.approx((12.0, -8.0), rel=1e-12)


@pytest.mark.parametrize("max_pairs", [1, 1000, 1 << 16])  # one row per block, ..., default
def test_blocked_evaluation_matches_single_block(monkeypatch, max_pairs):
    P, G, _ = polynomial_swallowtail(n_half=301)
    monkeypatch.setattr(intersect, "_MAX_PAIRS_PER_BLOCK", 1 << 30)
    single_block = find_equilibrium(P, G)

    monkeypatch.setattr(intersect, "_MAX_PAIRS_PER_BLOCK", max_pairs)

    assert find_equilibrium(P, G) == single_block


# --------------------------------------------------------------------------- #
# No crossing
# --------------------------------------------------------------------------- #

def test_supercritical_isotherm_has_no_crossing():
    P, G = vdw_isotherm(T=1.1, v=np.geomspace(0.45, 10.0, 2000))
    assert find_equilibrium(P, G) == (None, None)


def test_monotonic_branch_has_no_crossing():
    P = np.linspace(0.0, 10.0, 200)
    assert find_equilibrium(P, np.sqrt(P)) == (None, None)


def test_shared_vertex_of_adjacent_segments_is_not_a_crossing():
    # A sharp cusp: the two segments touch only at their common endpoint.
    assert find_equilibrium([0.0, 1.0, 0.0, -1.0], [0.0, 1.0, 2.0, 3.0]) == (None, None)


def test_semicubical_cusp_without_loop_has_no_crossing():
    s = np.linspace(-1.0, 1.0, 401)
    assert find_equilibrium(s**2, s**3) == (None, None)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_too_few_points_returns_none(n):
    assert find_equilibrium(np.arange(n), np.arange(n)) == (None, None)


# --------------------------------------------------------------------------- #
# Flat and parallel segments (rxs = 0); errstate turns any division by zero
# or invalid operation into an exception.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "P, G",
    [
        (np.linspace(0.0, 1.0, 50), np.zeros(50)),  # flat: all segments collinear
        (np.zeros(50), np.linspace(0.0, 1.0, 50)),  # constant pressure
        (np.ones(50), np.ones(50)),  # every point identical: zero-length segments
        ([0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 2, 2]),  # serpentine of parallel segments
        ([0, 2, 1, 3], [0, 0, 0, 0]),  # collinear overlap: no unique point
    ],
    ids=["flat", "isobaric", "single-point", "serpentine", "collinear-overlap"],
)
def test_parallel_segments_are_skipped_without_division_by_zero(P, G):
    with np.errstate(divide="raise", invalid="raise"):
        assert find_equilibrium(P, G) == (None, None)


def test_crossing_is_found_alongside_flat_and_repeated_segments():
    # The hand-built loop with a repeated point (zero-length segment) and flat
    # end segments that are collinear with each other.
    P = [-2.0, 0.0, 4.0, 4.0, 5.0, -1.0, 6.0, 8.0]
    G = [0.0, 0.0, 4.0, 4.0, 5.5, 3.5, 0.0, 0.0]
    with np.errstate(divide="raise", invalid="raise"):
        assert find_equilibrium(P, G) == pytest.approx((2.0, 2.0), rel=1e-12)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        find_equilibrium(np.arange(5.0), np.arange(6.0))


def test_multidimensional_input_raises():
    with pytest.raises(ValueError, match="1D"):
        find_equilibrium(np.zeros((3, 3)), np.zeros((3, 3)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_raises(bad):
    P = LOOP_P.copy()
    P[2] = bad
    with pytest.raises(ValueError, match="finite"):
        find_equilibrium(P, LOOP_G)
