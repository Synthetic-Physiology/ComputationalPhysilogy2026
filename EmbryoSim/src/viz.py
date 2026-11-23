"""Visualisierung für die Embryo-Simulation (Matplotlib, 2D)."""
from __future__ import annotations

import copy
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection, LineCollection, PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge

from simulation import FATE_NAMES, Fate, Simulation

# --------------------------------------------------------------------------
# Farbpalette & Rendering-Konstanten
# --------------------------------------------------------------------------
FATE_COLORS: dict[Fate, str] = {
    Fate.UNDECIDED: "#d9d9d9",  # hellgrau
    Fate.TE: "#4c9be8",         # blau
    Fate.ICM: "#e8704c",        # orange-rot
}

ZP_COLOR = "#c9a227"     # goldgelb (Zona pellucida)
LUMEN_COLOR = "#cfe8f7"  # hellblau (Blastozoel-Flüssigkeit)
LUMEN_EDGE = "#7fb2d4"   # Konturfarbe des Blastozoels
TJ_COLOR = "#163d6b"     # dunkelblau (Tight Junctions im TE-Ring)

_ZP_MIN_VISIBLE_FRAC = 0.12  # Minimale dargestellte Restdicke intakter ZP-Segmente
_METABALL_GRID = 72          # Gitterauflösung pro Achse für das Blastozoel-Metaball-Feld
_METABALL_PAD_SCALE = 3.0    # Randzugabe (in `fluid_r`) um die Partikel-Bounding-Box
_METABALL_SIGMA_SCALE = 1.45 # Gauß-Breite (in `fluid_r`) je Fluidpartikel
_METABALL_THRESH = 0.65      # Iso-Schwellwert für die glatte Flüssigkeitskontur


# --------------------------------------------------------------------------
# Interne Zeichen- und Layout-Helfer
# --------------------------------------------------------------------------
def _draw_zp(sim: Simulation, ax: Axes) -> None:
    """Zeichnet die Zona pellucida als Segmentring (offene Segmente fehlen).

    Hinweis: `Wedge(..., r_outer, ..., width=w)` zeichnet vom Außenradius nach innen.
    Abnehmende Festigkeit (`zp_I`) lässt die Zona daher visuell von der Innenseite her
    ausdünnen, passend zum lokalen Abbau durch den Embryo.
    """
    p = sim.p
    if not p.zp_enabled:
        return

    nseg = p.zp_segments
    r_outer = p.zp_radius + p.zp_thickness
    wedges: list[Wedge] = []
    for k in range(nseg):
        integrity = float(sim.zp_I[k])
        if integrity <= 0.0:
            continue
        width = p.zp_thickness * float(np.clip(integrity, _ZP_MIN_VISIBLE_FRAC, 1.0))
        a0 = np.degrees((k / nseg) * 2.0 * np.pi - np.pi)
        a1 = np.degrees(((k + 1) / nseg) * 2.0 * np.pi - np.pi)
        wedges.append(Wedge((0.0, 0.0), r_outer, a0, a1, width=width))

    if not wedges:
        return
    ax.add_collection(
        PatchCollection(wedges, facecolor=ZP_COLOR, edgecolor="none", alpha=0.85, zorder=0)
    )


def _draw_lumen(sim: Simulation, ax: Axes) -> None:
    """Zeichnet das Partikel-Blastozoel als glatte Flüssigkeitsmasse (Gauß-Metaball-Kontur)."""
    fp = sim.fluid_pos
    if len(fp) == 0:
        return

    r = sim.p.fluid_r
    pad = _METABALL_PAD_SCALE * r
    xmin, ymin = fp.min(axis=0) - pad
    xmax, ymax = fp.max(axis=0) + pad

    gx = np.linspace(xmin, xmax, _METABALL_GRID)
    gy = np.linspace(ymin, ymax, _METABALL_GRID)
    X, Y = np.meshgrid(gx, gy)

    sigma = _METABALL_SIGMA_SCALE * r
    dx = X[:, :, None] - fp[None, None, :, 0]
    dy = Y[:, :, None] - fp[None, None, :, 1]
    field = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma)).sum(axis=2)

    if field.max() >= _METABALL_THRESH:
        ax.contourf(
            X,
            Y,
            field,
            levels=[_METABALL_THRESH, float(field.max()) + 1.0],
            colors=[LUMEN_COLOR],
            alpha=0.92,
            zorder=1,
        )
        ax.contour(
            X,
            Y,
            field,
            levels=[_METABALL_THRESH],
            colors=[LUMEN_EDGE],
            linewidths=0.9,
            zorder=1,
        )


def _draw_tj(sim: Simulation, ax: Axes) -> None:
    """Zeichnet aktive Tight Junctions als Verbindungslinien zwischen gebundenen TE-Zellen."""
    bonds = sim.tj_bonds
    if len(bonds) == 0:
        return
    segs = np.stack([sim.pos[bonds[:, 0]], sim.pos[bonds[:, 1]]], axis=1)
    lc = LineCollection(segs, colors=TJ_COLOR, linewidths=1.6, alpha=0.9, zorder=3)
    ax.add_collection(lc)


def _auto_lim(sim: Simulation) -> float:
    """Berechnet die halbe Kantenlänge des Bildausschnitts."""
    if sim.p.zp_enabled:
        far = float(np.max(np.linalg.norm(sim.pos, axis=1) + sim.r))
        return 1.12 * max(sim.p.zp_radius + sim.p.zp_thickness, far)
    center = sim.pos.mean(axis=0)
    return 1.2 * float(np.max(np.abs(sim.pos - center) + sim.r[:, None])) + 0.1


def _center(sim: Simulation) -> np.ndarray:
    """Gibt das Bildzentrum zurück (Ursprung bei aktiver Zona, sonst Zellschwerpunkt)."""
    return np.zeros(2, dtype=float) if sim.p.zp_enabled else sim.pos.mean(axis=0)


def _handles(
    sims: Sequence[Simulation] | None = None,
    show_tj: bool = True,
) -> list[Patch | Line2D]:
    """Erstellt Legendeneinträge passend zu den aktivierten Modulen der Simulation(en)."""
    handles: list[Patch | Line2D] = [
        Patch(facecolor=FATE_COLORS[f], edgecolor="#333", label=FATE_NAMES[f])
        for f in FATE_NAMES
    ]

    if sims is None:
        has_tj = show_tj
        has_lumen = True
        has_zp = True
    else:
        has_tj = show_tj and any(s.p.tj_enabled for s in sims)
        has_lumen = any(s.p.lumen_enabled for s in sims)
        has_zp = any(s.p.zp_enabled for s in sims)

    if has_tj:
        handles.append(Line2D([0], [0], color=TJ_COLOR, lw=1.8, label="Tight Junctions (TE)"))
    if has_lumen:
        handles.append(Patch(facecolor=LUMEN_COLOR, edgecolor=LUMEN_EDGE, label="Blastozoel (Flüssigkeit)"))
    if has_zp:
        handles.append(Patch(facecolor=ZP_COLOR, edgecolor="none", label="Zona pellucida"))
    return handles


def _fmt_time(hours: float) -> str:
    """Formatiert die Zeit in Stunden und Tagen (z. B. 't = 132 h (5,5 d)')."""
    days = f"{hours / 24.0:.1f}".replace(".", ",")
    return f"t = {hours:.0f} h ({days} d)"


def _colors(sim: Simulation) -> list[str]:
    """Ordnet jeder Zelle die Farbe ihres aktuellen Schicksals zu."""
    return [FATE_COLORS[Fate(int(f))] for f in sim.fate]


# --------------------------------------------------------------------------
# Öffentliche Visualisierungsfunktionen
# --------------------------------------------------------------------------
def draw(
    sim: Simulation,
    ax: Axes | None = None,
    lim: float | None = None,
    title: str | None = None,
    legend: bool = True,
    show_tj: bool = True,
) -> Axes:
    """Zeichnet einen Simulationszustand. `lim` ist die halbe Kantenlänge (None = automatisch)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    center = _center(sim)
    if lim is None:
        lim = _auto_lim(sim)

    _draw_zp(sim, ax)
    _draw_lumen(sim, ax)

    coll = EllipseCollection(
        widths=2.0 * sim.r,
        heights=2.0 * sim.r,
        angles=0,
        units="xy",
        offsets=sim.pos.copy(),
        offset_transform=ax.transData,
        facecolors=_colors(sim),
        edgecolors="#333333",
        linewidths=0.6,
        zorder=2,
    )
    ax.add_collection(coll)

    if show_tj:
        _draw_tj(sim, ax)

    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title if title is not None else f"{_fmt_time(sim.t)}, N = {sim.n}", fontsize=10)

    if legend:
        handles = _handles([sim], show_tj=show_tj)
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            fontsize=7,
            frameon=False,
        )
    return ax


def timeline(
    sim: Simulation,
    times: Sequence[float],
    ncols: int = 4,
    size: float = 2.6,
    show_tj: bool = True,
) -> Figure:
    """Läuft die Simulation weiter und zeigt Momentaufnahmen zu den angegebenen Zeitpunkten (h).

    Hinweis: Verändert `sim` in-place (endet beim letzten Zeitpunkt in `times`).
    """
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(size * ncols, size * nrows + 0.3), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")

    for ax, t_end in zip(axes.ravel(), times):
        sim.run(max(0.0, t_end - sim.t))
        ax.axis("on")
        draw(sim, ax=ax, legend=False, show_tj=show_tj)

    handles = _handles([sim], show_tj=show_tj)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    return fig


def compare(
    sims: Sequence[Simulation],
    titles: Sequence[str] | None = None,
    size: float = 4.5,
    show_tj: bool = True,
) -> Figure:
    """Stellt mehrere Simulationen nebeneinander mit gemeinsamem Bildausschnitt dar."""
    lim = max(_auto_lim(s) for s in sims)
    fig, axes = plt.subplots(
        1, len(sims), figsize=(size * len(sims), size + 0.7), squeeze=False
    )
    for k, (ax, s) in enumerate(zip(axes.ravel(), sims)):
        prefix = f"{titles[k]}\n" if titles else ""
        draw(
            s,
            ax=ax,
            lim=lim,
            title=f"{prefix}{_fmt_time(s.t)}, N = {s.n}",
            legend=False,
            show_tj=show_tj,
        )

    handles = _handles(sims, show_tj=show_tj)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    return fig


def record(sim: Simulation, hours: float, every: float = 1.0) -> list[Simulation]:
    """Läuft `hours` weiter und speichert alle `every` Stunden eine Kopie des Zustands."""
    frames = [copy.deepcopy(sim)]
    n_chunks = int(round(hours / every))
    for _ in range(n_chunks):
        sim.run(every)
        frames.append(copy.deepcopy(sim))
    return frames


def make_gif(
    frames: Sequence[Simulation],
    path: str,
    fps: int = 12,
    size: float = 4.5,
    show_tj: bool = True,
) -> str:
    """Speichert eine Bildfolge als GIF (benötigt nur Pillow, kein ffmpeg)."""
    lim = max(_auto_lim(f) for f in frames)
    fig, ax = plt.subplots(figsize=(size, size + 0.4))

    def update(k: int) -> list:
        ax.clear()
        draw(frames[k], ax=ax, lim=lim, show_tj=show_tj)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
