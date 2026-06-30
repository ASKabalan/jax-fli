"""Private helper to resolve ts / nb_shells / density_widths into canonical lightcone geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.growth import Dplus_to_a, growth_factor

from ..utils import distances as _voronoi_distances

_VALID_SHELL_SPACINGS = ("comoving", "a", "growth", "equal_vol")


# ---------------------------------------------------------------------------
# Edge generators — all return ascending comoving distance
# ---------------------------------------------------------------------------


def _generate_edges_comoving(n, r_max):
    """Edges uniform in comoving distance."""
    return jnp.linspace(0.0, r_max, n + 1)


def _generate_edges_equal_vol(n, r_max, min_width=0.0):
    """Edges uniform in volume (r^3), with the outer shells floored to ``min_width``.

    Pure equal-volume places edges at ``r_i = r_max (i/n)**(1/3)``, so the
    outermost shells become arbitrarily thin.  When ``min_width`` binds, the
    outer shells are widened to ``min_width`` (greedily, from the far edge
    inward) and the remaining inner volume is re-partitioned equal-volume over
    the shells that are left.  Once an outer shell's natural width clears
    ``min_width``, every (fatter) inner shell clears it too, so the rest are laid
    down as one pure equal-volume partition.  With ``min_width`` below the natural
    outer width this is a no-op and reproduces the pure equal-volume edges.
    """
    # Count how many outer shells must be floored, peeling from the far edge.
    n_floor = 0
    remaining = float(r_max)
    while n_floor < n:
        k = n - n_floor  # shells not yet placed
        natural = remaining * (1.0 - ((k - 1) / k) ** (1.0 / 3.0))
        if natural >= min_width:
            break  # remaining shells are all >= natural >= min_width
        n_floor += 1
        remaining -= min_width

    # inner (n - n_floor) shells: equal-volume over [0, remaining];
    # outer n_floor shells: uniform min_width out to r_max.
    n_inner = n - n_floor
    inner_edges = remaining * (jnp.arange(n_inner + 1) / n_inner) ** (1.0 / 3.0) if n_inner > 0 else jnp.zeros(1)
    outer_edges = remaining + min_width * jnp.arange(1, n_floor + 1)
    return jnp.concatenate([inner_edges, outer_edges])


def _generate_edges_a(cosmo, n, r_max):
    """Edges uniform in scale factor.  *r_max* is comoving Mpc/h."""
    a_start = jc.background.a_of_chi(cosmo, jnp.array(r_max)).squeeze()
    a_edges = jnp.linspace(a_start, 1.0, n + 1)
    r_edges = jc.background.radial_comoving_distance(cosmo, a_edges)
    return r_edges[::-1]  # ascending r


def _generate_edges_growth(cosmo, n, r_max):
    """Edges uniform in growth factor D+(a).  *r_max* is comoving Mpc/h."""
    a_start = jc.background.a_of_chi(cosmo, jnp.array(r_max)).squeeze()
    D_start = growth_factor(cosmo, jnp.atleast_1d(a_start)).squeeze()
    D_today = growth_factor(cosmo, jnp.atleast_1d(1.0)).squeeze()
    D_edges = jnp.linspace(D_start, D_today, n + 1)
    a_edges = Dplus_to_a(cosmo, D_edges)
    r_edges = jc.background.radial_comoving_distance(cosmo, a_edges)
    return r_edges[::-1]  # ascending r


# ---------------------------------------------------------------------------
# Min-width checks (concrete values — function is NOT jittable)
# ---------------------------------------------------------------------------


def _check_min_width(widths, min_width):
    """Check min_width on concrete JAX array of widths."""
    min_w = float(jnp.min(widths))
    if min_w < min_width - 1e-10:
        raise ValueError(
            f"Minimum shell width {min_w:.2f} Mpc/h is below min_width={min_width} Mpc/h. "
            f"Reduce the number of shells/steps or lower min_width."
        )


def _check_min_width_static(shell_spacing, n, r_max, min_width):
    """Fast pure-Python check before computing edges (comoving / equal_vol only)."""
    if shell_spacing == "comoving":
        width = r_max / n
        if width < min_width - 1e-10:
            raise ValueError(
                f"Minimum shell width {width:.2f} Mpc/h is below min_width={min_width} Mpc/h. "
                f"Reduce the number of shells/steps or lower min_width."
            )
    elif shell_spacing == "equal_vol":
        # min_width reshapes the outer shells (hybrid equal-volume), so the only
        # failure is an outright-infeasible request: N shells of >= min_width
        # cannot tile [0, r_max].
        if n * min_width > r_max + 1e-10:
            raise ValueError(
                f"Cannot fit {n} equal_vol shells of width >= min_width={min_width} Mpc/h within "
                f"r_max={r_max:.2f} Mpc/h (need {n * min_width:.2f} Mpc/h). "
                f"Reduce the number of shells/steps or lower min_width."
            )


# ---------------------------------------------------------------------------
# nb_steps ↔ nb_shells extrapolation helpers
# ---------------------------------------------------------------------------


def compute_n_steps_for_shells(cosmo, shell_spacing, nb_shells, max_box_comoving, r_max_sim):
    """Given *nb_shells* inside the box, extrapolate total integration steps to *r_max_sim*.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology for distance calculations.
    shell_spacing : str
        One of ``'comoving'``, ``'a'``, ``'growth'``, ``'equal_vol'``.
    nb_shells : int
        Number of radial shells inside the box.
    max_box_comoving : float
        Maximum comoving distance of the box (half the box diagonal), in Mpc/h.
    r_max_sim : float
        Maximum simulation comoving distance in Mpc/h.

    Returns
    -------
    int
        Estimated total integration steps.
    """
    r_max_sim = float(r_max_sim)

    if shell_spacing == "comoving":
        delta = max_box_comoving / nb_shells
        steps_outside = int((r_max_sim - max_box_comoving) / delta)

    elif shell_spacing == "equal_vol":
        delta_r3 = max_box_comoving**3 / nb_shells
        steps_outside = int((r_max_sim**3 - max_box_comoving**3) / delta_r3)

    elif shell_spacing == "a":
        a_box = float(jc.background.a_of_chi(cosmo, jnp.array(max_box_comoving)).squeeze())
        a_ini = float(jc.background.a_of_chi(cosmo, jnp.array(r_max_sim)).squeeze())
        delta_a = (1.0 - a_box) / nb_shells
        steps_outside = int((a_box - a_ini) / delta_a)

    elif shell_spacing == "growth":
        a_box = float(jc.background.a_of_chi(cosmo, jnp.array(max_box_comoving)).squeeze())
        a_ini = float(jc.background.a_of_chi(cosmo, jnp.array(r_max_sim)).squeeze())
        D_box = float(growth_factor(cosmo, jnp.atleast_1d(a_box)).squeeze())
        D_ini = float(growth_factor(cosmo, jnp.atleast_1d(a_ini)).squeeze())
        D_today = float(growth_factor(cosmo, jnp.atleast_1d(1.0)).squeeze())
        delta_D = (D_today - D_box) / nb_shells
        steps_outside = int((D_box - D_ini) / delta_D)

    else:
        raise ValueError(f"Unknown shell_spacing={shell_spacing!r}.")

    return nb_shells + steps_outside


# ---------------------------------------------------------------------------
# Simulation stepping visualization
# ---------------------------------------------------------------------------


def simulation_stepping(
    cosmo,
    t0,
    t1,
    n_steps,
    *,
    ts=None,
    nb_shells=None,
    max_comoving_distance=None,
    shell_spacing="comoving",
    min_width=50.0,
    density_widths=None,
    time_stepping="a",
):
    """Return the exact scale factors visited during integration.

    Replicates the double-loop logic of ``integrate()``: the outer loop
    iterates over shell targets (from ``ts``), the inner loop advances by
    ``dt_internal`` in the solver's time variable, clipping to each target.

    When *nb_shells* and *max_comoving_distance* are provided, ``resolve_geometry``
    is called internally to derive the shell scale factors.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology for distance/growth calculations.
    t0 : float
        Initial scale factor.
    t1 : float
        Final scale factor.
    n_steps : int
        Number of integration steps from *t0* to *t1*.
    ts : array-like, optional
        Shell target scale factors (descending). Mutually exclusive with
        *nb_shells*.
    nb_shells : int, optional
        Number of lightcone shells. Requires *max_comoving_distance*.
    max_comoving_distance : float, optional
        Box half-diagonal in Mpc/h. Required when *nb_shells* is given.
    shell_spacing : str, default='comoving'
        Shell spacing mode (passed to ``resolve_geometry``).
    min_width : float, default=50.0
        Minimum shell width in Mpc/h (passed to ``resolve_geometry``).
    density_widths : array-like, optional
        Override shell widths (passed to ``resolve_geometry``).
    time_stepping : str, default='a'
        Integrator time variable: ``'a'``, ``'D'``, or ``'log_a'``.

    Returns
    -------
    jax.Array
        Scale factors at each visited time (including *t0* and shell targets).
    """
    t0, t1 = float(t0), float(t1)

    # Resolve shell targets if needed
    if nb_shells is not None:
        if max_comoving_distance is None:
            raise ValueError("`max_comoving_distance` is required when `nb_shells` is provided.")
        shell_ts, _, _ = resolve_geometry(
            cosmo,
            max_comoving_distance,
            nb_shells=nb_shells,
            shell_spacing=shell_spacing,
            min_width=min_width,
            density_widths=density_widths,
        )
        shell_targets = sorted(float(a) for a in shell_ts)
    elif ts is not None:
        shell_targets = sorted(float(a) for a in jnp.atleast_1d(jnp.asarray(ts)))
    else:
        shell_targets = None

    # Compute dt_internal in the solver's time variable
    dt_internal = _compute_dt(time_stepping, cosmo, t0, t1, n_steps)

    if shell_targets is None:
        # No shells — just step uniformly from t0 to t1
        a_vals = [t0]
        t_curr = t0
        for _ in range(n_steps):
            t_curr = _advance(time_stepping, cosmo, t_curr, dt_internal)
            a_vals.append(min(t_curr, t1))
        return jnp.array(a_vals)

    # Double-loop: outer over shell targets, inner stepping
    tol = 1e-10
    a_vals = [t0]
    t_curr = t0
    for t_target in shell_targets:
        t_target = float(t_target)
        while t_curr < t_target - tol:
            t_next = _advance(time_stepping, cosmo, t_curr, dt_internal)
            # Clip to target (mirrors _clip_to_end)
            if t_next > t_target - tol:
                t_next = t_target
            t_curr = t_next
            a_vals.append(t_curr)

    return jnp.array(a_vals)


def _compute_dt(time_stepping, cosmo, t0, t1, n_steps):
    """Compute step size in the solver's internal time variable (pure Python)."""
    if time_stepping == "a":
        return (t1 - t0) / n_steps
    elif time_stepping == "D":
        D0 = float(growth_factor(cosmo, jnp.atleast_1d(t0)).squeeze())
        D1 = float(growth_factor(cosmo, jnp.atleast_1d(t1)).squeeze())
        return (D1 - D0) / n_steps
    elif time_stepping == "log_a":
        return float((jnp.log(t1) - jnp.log(t0)) / n_steps)
    else:
        raise ValueError(f"Unknown time_stepping={time_stepping!r}. Use 'a', 'D', or 'log_a'.")


def _advance(time_stepping, cosmo, t_curr, dt_internal):
    """Advance t_curr by one step (pure Python)."""
    if time_stepping == "a":
        return t_curr + dt_internal
    elif time_stepping == "D":
        D_curr = float(growth_factor(cosmo, jnp.atleast_1d(t_curr)).squeeze())
        return float(Dplus_to_a(cosmo, jnp.array(D_curr + dt_internal)).squeeze())
    elif time_stepping == "log_a":
        import math

        return t_curr * math.exp(dt_internal)
    else:
        raise ValueError(f"Unknown time_stepping={time_stepping!r}.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resolve_geometry(
    cosmo,
    max_comoving_distance,
    *,
    ts=None,
    nb_shells=None,
    density_widths=None,
    shell_spacing: str = "comoving",
    min_width: float = 50.0,
    box_size_z: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolve the ``ts`` / ``nb_shells`` specification into canonical geometry.

    This function is **not jittable** — it evaluates cosmology functions eagerly
    to produce static Python ints for ``nb_shells``.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology for distance calculations.
    max_comoving_distance : float
        Maximum comoving distance of the box (half the box diagonal), in Mpc/h.
    ts : scalar, 1-D array, or 2-D ``(2, N)`` array, optional
        Scale-factor specification. Mutually exclusive with *nb_shells*.
    nb_shells : int, optional
        Number of radial shells inside the box. Mutually exclusive with *ts*.
    density_widths : scalar or array, optional
        Override shell widths.
    shell_spacing : str, default='comoving'
        ``'comoving'``, ``'a'``, ``'growth'``, or ``'equal_vol'``.
    min_width : float, default=50.0
        Minimum shell width in Mpc/h comoving.

    Returns
    -------
    ts : jax.Array
        Scale factors at shell centres.
    r_centers : jax.Array
        Comoving distances at shell centres.
    density_widths : jax.Array
        Shell widths in comoving distance.
    """
    if isinstance(max_comoving_distance, jax.core.Tracer):
        raise RuntimeError("resolve_geometry cannot be called with traced arguments. Call it outside jax.jit.")

    max_box_comoving = float(max_comoving_distance)

    # --- Validation ---
    if ts is not None and nb_shells is not None:
        raise ValueError("`ts` and `nb_shells` are mutually exclusive.")

    if ts is None and nb_shells is None:
        raise ValueError("At least one of `ts` or `nb_shells` must be provided.")

    if nb_shells is not None and shell_spacing not in _VALID_SHELL_SPACINGS:
        raise ValueError(f"shell_spacing={shell_spacing!r} not in {_VALID_SHELL_SPACINGS}.")

    # =====================================================================
    # Path A: Manual ts
    # =====================================================================
    if ts is not None:
        return _resolve_manual_ts(cosmo, ts, density_widths, max_box_comoving, box_size_z)

    # =====================================================================
    # Path B: Automatic generation from nb_shells
    # =====================================================================
    return _resolve_nb_shells(cosmo, shell_spacing, nb_shells, max_box_comoving, min_width, density_widths)


# ---------------------------------------------------------------------------
# Path A: Manual ts
# ---------------------------------------------------------------------------


def _resolve_manual_ts(cosmo, ts, density_widths, max_box_comoving, box_size_z):
    """Handle user-provided ts (scalar, 1-D, or 2-D)."""
    ts = jnp.asarray(ts)

    # --- scalar ts ---
    if ts.ndim == 0 or (ts.ndim == 1 and ts.size == 1):
        ts = jnp.atleast_1d(ts).squeeze()
        r_center = jc.background.radial_comoving_distance(cosmo, ts)
        if density_widths is not None:
            density_widths = jnp.atleast_1d(jnp.asarray(density_widths).squeeze())
        else:
            assert box_size_z is not None, "box_size_z is required when ts is scalar to determine density_widths."
            density_widths = jnp.array([box_size_z])
        return jnp.atleast_1d(ts), jnp.atleast_1d(r_center), density_widths

    # --- 1-D array: shell centres ---
    if ts.ndim == 1:
        indices_sorted = jnp.argsort(ts)
        ts = ts[indices_sorted]
        r_centers = jc.background.radial_comoving_distance(cosmo, ts)
        if density_widths is not None:
            try:
                density_widths = jnp.broadcast_to(density_widths, r_centers.shape)
                density_widths = density_widths[indices_sorted]
            except ValueError:
                raise ValueError("density_widths must be broadcastable to the shape of ts.")
        else:
            density_widths = _voronoi_distances(r_centers, max_box_comoving)
        return ts, r_centers, density_widths

    # --- 2-D array (2, N): near/far ---
    if ts.ndim == 2 and ts.shape[0] == 2:
        a_near, a_far = ts[0], ts[1]
        sorted_near_index = jnp.argsort(a_near)
        a_near = a_near[sorted_near_index]
        a_far = a_far[sorted_near_index]
        r_near, r_far = jc.background.radial_comoving_distance(cosmo, jnp.array([a_near, a_far]))
        r_centers = 0.5 * (r_near + r_far)
        ts_resolved = jc.background.a_of_chi(cosmo, r_centers)
        density_widths_out = jnp.abs(r_far - r_near)
        return ts_resolved, r_centers, density_widths_out

    raise ValueError(f"ts has unsupported shape {ts.shape}. Expected scalar, 1-D, or (2, N).")


# ---------------------------------------------------------------------------
# Path B: nb_shells — shells inside box only
# ---------------------------------------------------------------------------


def _resolve_nb_shells(cosmo, shell_spacing, nb_shells, max_box_comoving, min_width, density_widths):
    """Generate ``nb_shells`` shells inside the box."""
    # Generate edges inside box
    if shell_spacing == "comoving":
        _check_min_width_static("comoving", nb_shells, max_box_comoving, min_width)
        r_edges = _generate_edges_comoving(nb_shells, max_box_comoving)
    elif shell_spacing == "equal_vol":
        _check_min_width_static("equal_vol", nb_shells, max_box_comoving, min_width)
        r_edges = _generate_edges_equal_vol(nb_shells, max_box_comoving, min_width)
    elif shell_spacing == "a":
        r_edges = _generate_edges_a(cosmo, nb_shells, max_box_comoving)
    elif shell_spacing == "growth":
        r_edges = _generate_edges_growth(cosmo, nb_shells, max_box_comoving)
    else:
        raise ValueError(f"Unknown shell_spacing={shell_spacing!r}.")

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    widths = jnp.abs(r_edges[1:] - r_edges[:-1])

    # Min-width check for a/growth (comoving/equal_vol already caught above)
    if shell_spacing in ("a", "growth"):
        jax.debug.callback(_check_min_width, widths, min_width)

    if density_widths is not None:
        try:
            widths = jnp.broadcast_to(density_widths, r_centers.shape)
        except ValueError:
            raise ValueError("density_widths must be broadcastable to the shape of r_centers.")

    ts_resolved = jc.background.a_of_chi(cosmo, r_centers)

    # Sort descending in r (far-to-near)
    order = jnp.argsort(-r_centers)
    r_centers = r_centers[order]
    ts_resolved = ts_resolved[order]
    widths = widths[order]

    return ts_resolved, r_centers, widths
