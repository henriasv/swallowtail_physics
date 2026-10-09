# swallowtail_physics

Find the phase-equilibrium point of a first-order transition from Gibbs free
energy data, by locating where the swallowtail loop in G(P) crosses itself.

Below the critical temperature, G(P) along an isotherm of a van der Waals-like
equation of state folds into a swallowtail: two stable branches joined by an
unstable branch with a cusp at each spinodal. The stable branches cross where
the two phases have equal Gibbs free energy. That crossing gives the coexistence
pressure, and it is equivalent to Maxwell's equal-area construction.

## Installation

Requires Python ≥ 3.9 and NumPy (installed automatically).

```bash
pip install git+https://github.com/henriasv/swallowtail_physics.git
```

For development, with the test dependencies:

```bash
git clone https://github.com/henriasv/swallowtail_physics.git
cd swallowtail_physics
pip install -e ".[test]"
pytest
```

## Usage

```python
import numpy as np
from swallowtail_physics import find_equilibrium

# Reduced van der Waals isotherm below the critical temperature
T = 0.9
v = np.geomspace(0.45, 10.0, 5000)
P = 8 * T / (3 * v - 1) - 3 / v**2
G = -(8 * T / 3) * np.log(3 * v - 1) - 3 / v + P * v

P_eq, G_eq = find_equilibrium(P, G)
print(f"P_eq = {P_eq:.4f}, G_eq = {G_eq:.4f}")  # P_eq = 0.6470, G_eq = -4.0763
```

`find_equilibrium(P, G)` takes two 1D arrays ordered along the curve and
returns `(P_eq, G_eq)` as floats, or `(None, None)` if the curve does not
cross itself (for example above the critical temperature).

### Volume change across the transition

Pass the volume array as well to get the volume of each coexisting phase:

```python
from swallowtail_physics import find_coexistence

result = find_coexistence(P, G, v)
print(f"V1 = {result.V1:.4f}, V2 = {result.V2:.4f}, dV = {result.dV:.4f}")
# V1 = 0.6034, V2 = 2.3488, dV = 1.7454
```

`find_coexistence(P, G, V)` finds the crossing in G(P) exactly like
`find_equilibrium` and returns a named tuple `(P, G, V1, V2)` with
`dV = V2 - V1`, or `None` if there is no crossing. The crossing point lies on
one segment of each stable branch, and `V` is interpolated at that point on
both segments. `V1` belongs to the branch reached first along the data, so `dV`
is positive when the data are ordered by increasing volume.

Since (∂G/∂P)_T = V, `dV` is the jump in slope of G(P) at the transition.
Interpolating V is second-order accurate in the point spacing. Estimating `dV`
from the segment slopes instead is only first-order: at 2000 points it is about
1000 times less accurate for the van der Waals fluid. With `dV` at several
temperatures, the Clausius–Clapeyron relation gives the latent heat,
`L = T · dV · dP_eq/dT`.

## How it works

The data are treated as a polyline of segments `p_i + t r_i`, `t ∈ [0, 1]`.
For each pair of segments the intersection parameters follow from 2D cross
products, with `q = p_j - p_i`:

```
t = (q × r_j) / (r_i × r_j),    u = (q × r_i) / (r_i × r_j)
```

and the segments intersect when `0 ≤ t, u ≤ 1`.

- **Vectorized.** All pairs `j ≥ i + 2` are evaluated at once by broadcasting
  segment arrays into `(i, j)` matrices. There is no Python loop over segments.
- **Adjacent segments ignored.** Segments with `|i - j| ≤ 1` share an endpoint
  and are never reported as a crossing.
- **No division by zero.** The range test is done without dividing (multiplying
  through by the sign of `r_i × r_j`). Parallel, collinear and zero-length
  segments are masked out before the single division on the final hits, so
  flat regions and repeated points are safe.
- **Unit independent.** P and G are rescaled to [0, 1] before the geometry, so
  the parallel tolerance does not depend on whether P is in Pa or bar. `t` and
  `u` are invariant under this map and are applied to the original data.
- **Cache-sized blocks.** The pair grid is broadcast in row blocks of about
  65 000 pairs. That keeps memory bounded for any N, runs about twice as fast
  as one big N² broadcast, and lets the search stop at the first block that
  contains a crossing. A full scan of 5000 points (12.5 M segment pairs) takes
  about 90 ms on an Apple-silicon laptop.

If the data cross themselves more than once, the crossing reached first when
walking along the arrays is returned. A collinear overlap has no unique
intersection point and is not reported.

## Tests

```bash
pytest
```

The suite covers exact crossings on programmatically generated swallowtails,
convergence to an analytic crossing, coexistence pressure and phase volumes
checked against an independent Maxwell construction for the van der Waals
fluid, inputs without a crossing, flat and
parallel segments, and input validation. NumPy warnings are treated as errors.

## License

MIT. See [LICENSE](LICENSE).
