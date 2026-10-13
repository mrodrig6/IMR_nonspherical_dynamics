"""Draw the reconstructed fields.

Everything physical lives in :mod:`imrns.postprocess.fields`; this module only
turns arrays into figures.  Matplotlib is imported lazily so the solver does
not depend on it.

Three entry points:

* :func:`field_slice` -- evaluate a scalar field on a meridional plane and
  return ``(X, Z, C)``, ready for ``pcolormesh``.  Use it to build your own
  figure.
* :func:`plot_frame` -- draw one frame onto an axis.
* :func:`snapshot_grid` and :func:`animate` -- a panel grid and a movie,
  replacing the two output modes of the MATLAB original.

Colour conventions
------------------
Signed fields (every strain, rate and shear-stress component) are drawn with a
diverging map on limits symmetric about zero, so the midpoint is zero and equal
magnitudes of either sign read as equally intense.  That symmetry is a property
of the colormap as well as the limits; see :func:`imrns.colormaps.brown_centered_diverging`.
Unsigned fields (``frobenius``, ``maxabs_component``) get a sequential map from
zero instead, because a diverging map on a non-negative field wastes half its
range and implies a polarity that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..colormaps import get as get_colormap
from .fields import (build_modal_field, rate_of_strain_components, strain_components,
                     stress_components, tensor_scalar)

__all__ = ["FieldSpec", "field_slice", "plot_frame", "snapshot_grid", "animate",
           "pick_frames", "symmetric_limits"]

UNSIGNED_SCALARS = {"frobenius", "norm", "magnitude", "maxabs_component",
                    "max_abs_component", "component_max"}

_LABELS = {
    "almansi": "Almansi strain",
    "small": "infinitesimal strain",
    "strain_rate": "rate of strain",
    "stress": "Cauchy stress",
}
_COMPONENT_LABELS = {"err": "rr", "ett": r"\theta\theta", "epp": r"\phi\phi",
                     "ert": r"r\theta", "erp": r"r\phi", "etp": r"\theta\phi"}


@dataclass
class FieldSpec:
    """What to plot and with which material parameters.

    ``measure`` is ``"almansi"``, ``"small"``, ``"strain_rate"`` or
    ``"stress"``; ``scalar`` selects a component or invariant (see
    :func:`imrns.postprocess.fields.tensor_scalar`).  ``G``, ``mu``, ``alpha``
    and ``pressure`` are used only by ``"stress"``.
    """

    measure: str = "almansi"
    scalar: str = "ert"
    Rref: float = 1.0
    angular_mode: str = "axisymmetric"
    harmonic_part: str = "cos"
    G: float = 1.0
    mu: float = 0.0
    alpha: float = 0.0
    pressure: float | None = None

    @property
    def needs_rates(self) -> bool:
        return self.measure in ("strain_rate", "stress")

    def label(self) -> str:
        """A LaTeX axis label for this field."""
        sym = {"almansi": "e", "small": r"\varepsilon", "strain_rate": "D", "stress": r"\sigma"}
        base = sym.get(self.measure, "f")
        comp = _COMPONENT_LABELS.get(self.scalar.lower())
        if comp:
            return rf"${base}_{{{comp}}}$"
        return f"{_LABELS.get(self.measure, self.measure)} ({self.scalar})"


def _components(field, r_grid, theta, spec: FieldSpec, az=None):
    if spec.measure in ("almansi", "small"):
        return strain_components(field, r_grid, theta, az, spec.Rref, spec.measure,
                                 spec.angular_mode, spec.harmonic_part)
    if spec.measure == "strain_rate":
        return rate_of_strain_components(field, r_grid, theta, az,
                                         spec.angular_mode, spec.harmonic_part)
    if spec.measure == "stress":
        return stress_components(field, r_grid, theta, az, spec.Rref, spec.G, spec.mu,
                                 spec.alpha, spec.pressure, spec.angular_mode,
                                 spec.harmonic_part)
    raise ValueError(f"unknown measure {spec.measure!r}")


def wall_radius(theta, ep_now, R, modes, spec: FieldSpec, az=None):
    """Perturbed bubble wall ``r_b(theta) = R + sum_n eps_n Y_n``."""
    from .fields import harmonics

    rb = np.full_like(np.asarray(theta, dtype=float), float(R))
    for j, n in enumerate(modes):
        Y = harmonics(int(n), theta, az, mode=spec.angular_mode, part=spec.harmonic_part)[0]
        rb = rb + ep_now[j] * Y
    return rb


def _auto_r_max(result, frame, r_max):
    """Default the outer radius to four wall radii.

    The perturbation decays like a power of ``r``, so a fixed outer radius
    leaves most of the panel empty once the bubble is small.
    """
    return 4.0 * float(result.R[frame]) if r_max is None else float(r_max)


def field_slice(result, frame: int, L: float, spec: FieldSpec | None = None, *,
                r_max: float | None = None, n_radial: int = 160, n_theta: int = 240,
                wall_clearance: float = 1e-3):
    """Evaluate the scalar field on a meridional slice.

    Returns ``(X, Z, C)`` in the plane ``phi = 0``, where ``X`` is the
    cylindrical radius (signed, so the full circle is drawn) and ``Z`` the axis
    of symmetry.  Pass them straight to ``pcolormesh``.

    The radial grid starts just outside the *perturbed* wall at each angle, so
    the field is never evaluated inside the bubble.  ``r_max`` is in the same
    units as ``R``.
    """
    spec = spec or FieldSpec()
    r_max = _auto_r_max(result, frame, r_max)
    field = build_modal_field(result, frame, L, spec.Rref, with_rates=spec.needs_rates)

    theta = np.linspace(0.0, np.pi, n_theta)
    ep_now = result.ep[:, frame]
    rb = wall_radius(theta, ep_now, field.R, result.modes, spec)
    rb = np.maximum(rb, 1e-6)

    # One radial ray per angle, each starting at that angle's wall.
    s = np.linspace(0.0, 1.0, n_radial)[:, None]
    start = rb[None, :] * (1.0 + wall_clearance)
    r_grid = start + s * (r_max - start)
    theta_grid = np.broadcast_to(theta[None, :], r_grid.shape)

    C = tensor_scalar(_components(field, r_grid, theta_grid, spec), spec.scalar)

    X = r_grid * np.sin(theta_grid)
    Z = r_grid * np.cos(theta_grid)
    return X, Z, np.asarray(C, dtype=float)


def symmetric_limits(C, quantile: float = 0.995):
    """Colour limits symmetric about zero, robust to a few extreme cells.

    Near-wall cells can be an order of magnitude above the rest of the field;
    taking the raw max would flatten everything else to the midpoint.
    """
    finite = np.asarray(C)[np.isfinite(C)]
    if finite.size == 0:
        return (-1.0, 1.0)
    m = float(np.quantile(np.abs(finite), quantile))
    if m <= 0:
        m = float(np.abs(finite).max()) or 1.0
    return (-m, m)


def pick_frames(result, n_frames: int, time_range=None):
    """Evenly spaced frame indices over ``time_range`` (default: the whole run)."""
    t = result.t
    lo, hi = (t[0], t[-1]) if time_range is None else time_range
    targets = np.linspace(lo, hi, int(n_frames))
    return [int(np.argmin(np.abs(t - q))) for q in targets]


def plot_frame(ax, result, frame: int, L: float, spec: FieldSpec | None = None, *,
               clim=None, cmap="brown_centered_diverging", r_max: float | None = None,
               n_radial: int = 160, n_theta: int = 240, show_wall: bool = True,
               wall_color="white", wall_width: float = 1.2, title: str | None = None):
    """Draw one frame onto ``ax``.  Returns the ``QuadMesh``."""
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    spec = spec or FieldSpec()
    r_max = _auto_r_max(result, frame, r_max)
    X, Z, C = field_slice(result, frame, L, spec, r_max=r_max,
                          n_radial=n_radial, n_theta=n_theta)

    unsigned = spec.scalar.lower() in UNSIGNED_SCALARS
    if clim is None:
        clim = (0.0, symmetric_limits(C)[1]) if unsigned else symmetric_limits(C)
    norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    cmap_obj = ListedColormap(get_colormap("viridis" if unsigned else cmap, 256))

    # Mirror across the axis of symmetry so the full meridional section shows.
    mesh = ax.pcolormesh(np.hstack([-X[:, ::-1], X]), np.hstack([Z[:, ::-1], Z]),
                         np.hstack([C[:, ::-1], C]), cmap=cmap_obj, norm=norm,
                         shading="gouraud", rasterized=True)

    if show_wall:
        theta = np.linspace(0.0, 2 * np.pi, 400)
        th_m = np.where(theta <= np.pi, theta, 2 * np.pi - theta)
        rb = wall_radius(th_m, result.ep[:, frame], result.R[frame], result.modes, spec)
        ax.plot(rb * np.sin(theta), rb * np.cos(theta), color=wall_color,
                lw=wall_width, solid_joinstyle="round")

    ax.set_aspect("equal")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=10)
    return mesh


def _shared_limits(result, frames, L, spec, r_max, n_radial, n_theta):
    """Colour limits covering every frame, so panels are comparable."""
    peak = 0.0
    for fr in frames:
        _, _, C = field_slice(result, fr, L, spec, r_max=r_max,
                              n_radial=n_radial, n_theta=n_theta)
        peak = max(peak, abs(symmetric_limits(C)[1]))
    return peak or 1.0


def snapshot_grid(result, L: float, output, spec: FieldSpec | None = None, *,
                  frames=None, n_frames: int = 12, layout=None, clim=None,
                  cmap="brown_centered_diverging", r_max: float | None = None,
                  n_radial: int = 160, n_theta: int = 240, time_range=None,
                  figsize=None, dpi: int = 300, colorbar_label: str | None = None,
                  panel_labels: bool = True):
    """Write a grid of snapshots to ``output`` (any format matplotlib knows).

    All panels share one colour scale, computed over the frames shown, so they
    can be compared directly.  Returns the figure.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from string import ascii_lowercase

    spec = spec or FieldSpec()
    frames = list(frames) if frames is not None else pick_frames(result, n_frames, time_range)
    if layout is None:
        n_cols = int(np.ceil(np.sqrt(len(frames))))
        n_rows = int(np.ceil(len(frames) / n_cols))
    else:
        n_rows, n_cols = layout

    if r_max is None:
        # One outer radius for every panel, so the panels are comparable.
        r_max = 4.0 * float(np.max(result.R[frames]))
    if clim is None:
        peak = _shared_limits(result, frames, L, spec, r_max, n_radial, n_theta)
        unsigned = spec.scalar.lower() in UNSIGNED_SCALARS
        clim = (0.0, peak) if unsigned else (-peak, peak)

    figsize = figsize or (2.6 * n_cols, 2.6 * n_rows + 0.7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for k, ax in enumerate(axes):
        if k >= len(frames):
            ax.axis("off")
            continue
        fr = frames[k]
        mesh = plot_frame(ax, result, fr, L, spec, clim=clim, cmap=cmap, r_max=r_max,
                          n_radial=n_radial, n_theta=n_theta,
                          title=rf"$t^* = {result.t[fr]:.3g}$")
        if panel_labels and k < len(ascii_lowercase):
            ax.text(0.02, 0.98, f"({ascii_lowercase[k]})", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, color="0.25")

    cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.85, aspect=30, pad=0.02)
    cbar.set_label(colorbar_label or spec.label(), fontsize=11)
    cbar.outline.set_visible(False)

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return fig


def animate(result, L: float, output, spec: FieldSpec | None = None, *,
            frames=None, n_frames: int = 120, fps: int = 20, clim=None,
            cmap="brown_centered_diverging", r_max: float | None = None,
            n_radial: int = 160, n_theta: int = 240, time_range=None,
            figsize=(5.0, 5.0), dpi: int = 150, colorbar_label: str | None = None):
    """Write a movie to ``output`` (``.mp4`` needs ffmpeg; ``.gif`` does not).

    The colour scale is fixed across the whole animation, so brightness changes
    mean the field changed and not that the scale moved.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    spec = spec or FieldSpec()
    frames = list(frames) if frames is not None else pick_frames(result, n_frames, time_range)
    if r_max is None:
        r_max = 4.0 * float(np.max(result.R[frames]))

    if clim is None:
        probe = frames[:: max(1, len(frames) // 12)]
        peak = _shared_limits(result, probe, L, spec, r_max, n_radial, n_theta)
        unsigned = spec.scalar.lower() in UNSIGNED_SCALARS
        clim = (0.0, peak) if unsigned else (-peak, peak)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    mesh = plot_frame(ax, result, frames[0], L, spec, clim=clim, cmap=cmap, r_max=r_max,
                      n_radial=n_radial, n_theta=n_theta,
                      title=rf"$t^* = {result.t[frames[0]]:.3g}$")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, aspect=30, pad=0.02)
    cbar.set_label(colorbar_label or spec.label(), fontsize=11)
    cbar.outline.set_visible(False)

    def update(fr):
        ax.clear()
        plot_frame(ax, result, fr, L, spec, clim=clim, cmap=cmap, r_max=r_max,
                   n_radial=n_radial, n_theta=n_theta,
                   title=rf"$t^* = {result.t[fr]:.3g}$")
        return ()

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = str(output)
    writer = PillowWriter(fps=fps) if out.lower().endswith(".gif") else FFMpegWriter(fps=fps)
    anim.save(out, writer=writer, dpi=dpi)
    plt.close(fig)
    return out
