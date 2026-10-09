"""Self-intersection of a swallowtail loop in Gibbs free energy data.

Below the critical temperature, G(P) traced along an isotherm of a van der
Waals-like equation of state folds back on itself and forms a swallowtail: two
locally stable branches joined by an unstable branch with a cusp at each
spinodal. The two stable branches cross where their Gibbs free energies are
equal, which is the phase-coexistence point (equivalent to Maxwell's equal-area
construction).

The data are treated as a polyline of segments p_i -> p_i + r_i. Two segments
meet where

    p_i + t r_i = p_j + u r_j,

and crossing this with r_j and r_i gives, with q = p_j - p_i and the 2D cross
product a x b = a_x b_y - a_y b_x,

    t = (q x r_j) / (r_i x r_j),    u = (q x r_i) / (r_i x r_j).

The segments intersect iff 0 <= t <= 1 and 0 <= u <= 1. All segment pairs are
evaluated at once by broadcasting over an (i, j) grid.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Coexistence", "find_coexistence", "find_equilibrium"]

# Segment pairs with |r_i x r_j| <= _PARALLEL_RTOL * |r_i| |r_j|, i.e. with
# |sin(angle)| below this, are treated as parallel and skipped. Zero-length
# segments from repeated points fall in this class too.
_PARALLEL_RTOL = 1e-12

# Upper bound on segment pairs evaluated per broadcast block. Keeping each
# block's temporaries cache-sized (~0.5 MB) was measured to be about twice as
# fast as one large N**2 broadcast, bounds memory for any N, and lets the search
# stop at the first block that contains a crossing.
_MAX_PAIRS_PER_BLOCK = 1 << 16


class Coexistence(NamedTuple):
    """Phase-coexistence point found at a swallowtail crossing.

    ``V1`` is the volume on the branch reached first along the data and ``V2``
    the volume on the branch reached second.
    """

    P: float
    G: float
    V1: float
    V2: float

    @property
    def dV(self) -> float:
        """Volume change ``V2 - V1`` across the transition."""
        return self.V2 - self.V1


def find_equilibrium(
    P: ArrayLike, G: ArrayLike
) -> Union[Tuple[float, float], Tuple[None, None]]:
    """Find the self-intersection of a swallowtail loop in G(P) data.

    Parameters
    ----------
    P, G : array_like, shape (N,)
        Pressure and Gibbs free energy, ordered along the curve (for example by
        increasing molar volume along an isotherm). Any consistent units.

    Returns
    -------
    (P_eq, G_eq) : tuple of float, or (None, None)
        Coordinates of the self-intersection, linearly interpolated on the
        crossing segments. ``(None, None)`` if the curve does not cross itself.

    Raises
    ------
    ValueError
        If ``P`` and ``G`` are not 1D, differ in length, or contain non-finite
        values.

    Notes
    -----
    Adjacent segments (``abs(i - j) <= 1``) share an endpoint and are never
    reported as crossing. Parallel and collinear segment pairs are skipped, so
    flat regions and repeated points are safe; a collinear overlap has no unique
    intersection point and is not reported. If the curve crosses itself more
    than once, the crossing reached first when walking along the data is
    returned.

    The search is O(N**2) in time over all segment pairs and fully vectorized;
    memory is bounded by evaluating the pair grid in row blocks.

    See Also
    --------
    find_coexistence : Also returns the volumes of both phases.
    """
    P, G = _as_curve(P, G, names="P and G")
    crossing = _crossing(P, G)
    if crossing is None:
        return None, None

    i, t, _, _ = crossing
    return float(_lerp(P, i, t)), float(_lerp(G, i, t))


def find_coexistence(P: ArrayLike, G: ArrayLike, V: ArrayLike) -> Optional[Coexistence]:
    """Find the coexistence point and the volume of each phase.

    The crossing is located in G(P) exactly as in :func:`find_equilibrium`.
    The crossing point lies on one segment of each stable branch, and ``V`` is
    interpolated at that point on both segments, giving the volumes of the two
    coexisting phases.

    Parameters
    ----------
    P, G : array_like, shape (N,)
        Pressure and Gibbs free energy, ordered along the curve.
    V : array_like, shape (N,)
        Volume (e.g. molar volume) at each data point, in any units.

    Returns
    -------
    Coexistence or None
        Named tuple ``(P, G, V1, V2)`` with the volume change available as
        ``.dV = V2 - V1``. ``V1`` belongs to the branch reached first along the
        data, so ``dV`` is positive when the data are ordered by increasing
        volume. ``None`` if the curve does not cross itself.

    Raises
    ------
    ValueError
        If ``P``, ``G`` and ``V`` are not 1D, differ in length, or contain
        non-finite values.

    Notes
    -----
    Because (dG/dP)_T = V, ``dV`` is the jump in slope of the stable G(P)
    envelope at the transition. Interpolating ``V`` is second-order accurate
    in the point spacing, whereas differencing segment slopes is first-order.
    """
    P, G, V = _as_curve(P, G, V, names="P, G and V")
    crossing = _crossing(P, G)
    if crossing is None:
        return None

    i, t, j, u = crossing
    return Coexistence(
        P=float(_lerp(P, i, t)),
        G=float(_lerp(G, i, t)),
        V1=float(_lerp(V, i, t)),
        V2=float(_lerp(V, j, u)),
    )


def _as_curve(*arrays: ArrayLike, names: str) -> Tuple[np.ndarray, ...]:
    out = tuple(np.asarray(a, dtype=float) for a in arrays)
    if any(a.ndim != 1 for a in out):
        shapes = ", ".join(str(a.shape) for a in out)
        raise ValueError(f"{names} must be 1D arrays, got shapes {shapes}.")
    if len({a.shape for a in out}) > 1:
        sizes = ", ".join(str(a.size) for a in out)
        raise ValueError(f"{names} must have the same length, got {sizes}.")
    if not all(np.isfinite(a).all() for a in out):
        raise ValueError(f"{names} must contain only finite values.")
    return out


def _lerp(x: np.ndarray, k: int, s: float) -> float:
    """Value of ``x`` at fraction ``s`` along segment ``k``."""
    return x[k] + s * (x[k + 1] - x[k])


def _crossing(P: np.ndarray, G: np.ndarray) -> Optional[Tuple[int, float, int, float]]:
    if P.size < 4:  # need at least two non-adjacent segments
        return None
    return _first_crossing(_rescale(P), _rescale(G))


def _rescale(x: np.ndarray) -> np.ndarray:
    """Map ``x`` onto [0, 1].

    Pressure and free energy typically differ by orders of magnitude (Pa vs
    J/mol), which would make the parallel test depend on the choice of units.
    The intersection parameters t and u are invariant under this affine map,
    so they can be computed on the rescaled curve and applied to the original.
    """
    shifted = x - x.min()
    span = shifted.max()
    return shifted / span if span > 0 else shifted


def _cross(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """2D cross product a x b = a_x b_y - a_y b_x, broadcast elementwise."""
    return ax * by - ay * bx


def _first_crossing(
    x: np.ndarray, y: np.ndarray
) -> Optional[Tuple[int, float, int, float]]:
    """Return ``(i, t, j, u)`` of the first crossing along the polyline, or None.

    The crossing lies at ``t`` along segment ``i`` and at ``u`` along segment
    ``j``, where ``i < j``.
    """
    px, py = x[:-1], y[:-1]  # segment start points p, shape (M,)
    rx, ry = np.diff(x), np.diff(y)  # segment vectors r, shape (M,)
    r_len = np.hypot(rx, ry)
    m = rx.size

    # Only pairs j >= i + 2 are needed: j < i is the same pair mirrored, and
    # j = i + 1 is adjacent. Rows i are processed in blocks, each against
    # columns j >= a + 2: column vectors (B, 1) broadcast against row vectors
    # (1, C) give (B, C) matrices over the pair grid. x and y components are
    # kept in separate arrays so every operation runs on contiguous memory.
    block = max(1, _MAX_PAIRS_PER_BLOCK // m)
    for a in range(0, m - 2, block):
        b = min(a + block, m - 2)
        rows = slice(a, b)
        cols = slice(a + 2, m)

        rx_i, ry_i = rx[rows, None], ry[rows, None]  # (B, 1)
        rx_j, ry_j = rx[None, cols], ry[None, cols]  # (1, C)
        qx = px[None, cols] - px[rows, None]  # q = p_j - p_i, (B, C)
        qy = py[None, cols] - py[rows, None]

        rxs = _cross(rx_i, ry_i, rx_j, ry_j)
        t_num = _cross(qx, qy, rx_j, ry_j)
        u_num = _cross(qx, qy, rx_i, ry_i)

        # 0 <= t_num / rxs <= 1 is tested without dividing: multiplying through
        # by sign(rxs) leaves 0 <= t_num * sign <= |rxs|. Parallel pairs
        # (rxs ~ 0) are masked out, so no division by zero ever happens.
        sign = np.sign(rxs)
        denom = np.abs(rxs)
        t_num *= sign
        u_num *= sign

        i_idx = np.arange(a, b)[:, None]
        j_idx = np.arange(a + 2, m)[None, :]
        hit = (
            (j_idx - i_idx >= 2)
            & (denom > _PARALLEL_RTOL * r_len[rows, None] * r_len[None, cols])
            & (t_num >= 0.0)
            & (t_num <= denom)
            & (u_num >= 0.0)
            & (u_num <= denom)
        )

        if hit.any():
            bi, bj = np.nonzero(hit)
            t = t_num[bi, bj] / denom[bi, bj]
            # Blocks run in order of i, so the earliest point along the curve
            # (smallest i + t) in the first block with a hit is the global one.
            k = np.argmin(bi + t)
            bi, bj = bi[k], bj[k]
            u = u_num[bi, bj] / denom[bi, bj]
            return a + int(bi), float(t[k]), a + 2 + int(bj), float(u)

    return None
