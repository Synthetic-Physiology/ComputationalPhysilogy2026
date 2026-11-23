"""
Embryo-Sim, Stufe 1: Zygote -> Morula -> Blastozyste in der Zona pellucida (2D, agentenbasiert)

Das Modell ist ein 2D-ÄQUATORIALSCHNITT: Zellzahlen sind 2D-Äquivalente (ein Schnitt zeigt
nur einen Bruchteil der Zellen einer echten Blastozyste). Zeit in "Stunden" nur grob kalibriert.

Regeln (bewusst heuristisch):
  * Zellen sind weiche Kreise (Position, Radius) mit schicksalsabhängiger Abstoßung und
    kurzreichweitiger Adhäsion.
  * Teilung nach individueller Zellzyklusdauer; in der Furchung ohne Wachstum (Töchter kleiner).
  * Schicksal TE (außen) / ICM (innen): freie Oberfläche -> Trophoblast, umringt -> Embryoblast
    (Stellvertreter für Polarität/Hippo-YAP). Bis `lock_n` plastisch, danach endgültig.
  * Trophoblast teilt sich tangential zur Oberfläche (Epithel) und bildet nach dem Lock-in
    laterale Tight Junctions aus, die den TE-Ring als geschlossenes Epithel zusammenhalten.
  * Blastozoel (Partikel-Fluid): Nach dem Lock-in pumpen Trophoblastzellen (basolateral)
    Fluidpartikel in das Embryo-Innere (bis zu einem Ceiling `fluid_max`, nach dem Hatching
    mit reduzierter Rate). Die Partikel stoßen sich untereinander ab (hydrostatischer Druck),
    verschmelzen durch leichte Kohäsion (Oberflächenspannung) zu einer zusammenhängenden Höhle
    und laufen bei größeren Lücken im TE-Ring realistisch aus.
  * Zona pellucida: Ring aus Segmenten; Abbau durch lytische Enzyme (TE im Wandkontakt) und
    durch Druck (Überlappung der Zellen mit der Wand). Rückkopplung -> ein einzelnes Loch.

NICHT enthalten (bewusst): 3D, echte Genregulation, aktive Zellmigration, Implantation.

Alle Zustände liegen in NumPy-Arrays (Structure-of-Arrays statt Agenten-Objekte),
damit der Code schnell bleibt und später leicht auf Numba umgestellt werden kann.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Numerische und geometrische Modellkonstanten
# --------------------------------------------------------------------------
_EPS = 1e-9                   # Schutz gegen Division durch Null
_LUMEN_PACKING_FACTOR = 1.15  # 2D-Flächenfaktor (Scheiben + Zwickel) für `lumen_area`
_ZP_ENZYME_MARGIN = 0.05      # Zusätzliche Kontaktreichweite für TE-Enzymwirkung an der ZP
_DIV_OFFSET_FACTOR = 0.90     # Tochterabstand bei Teilung als Anteil von `r_new`
_FLUID_SPAWN_OFFSET = 0.60    # Abstandsfaktor (relativ zu `fluid_r`) beim Pumpen neuer Partikel
_ICM_PUMP_BLOCK_FACTOR = 0.90 # Sperrradius-Faktor: polares TE pumpt nicht direkt ins ICM hinein
_LEAKAGE_SECTORS = 16         # Winkelauflösung für die emergente Fluid-Leckage


# --------------------------------------------------------------------------
# Zellschicksale (IntEnum für kompakte np.int8-Arrays, leicht erweiterbar)
# --------------------------------------------------------------------------
class Fate(IntEnum):
    """Integer-Codes der Zellschicksale (erweiterbar um z. B. EPI / HYP)."""

    UNDECIDED = 0
    TE = 1
    ICM = 2


# Abwärtskompatible Modul-Aliase (z. B. `from simulation import TE, ICM`)
UNDECIDED = Fate.UNDECIDED
TE = Fate.TE
ICM = Fate.ICM

FATE_NAMES: dict[Fate, str] = {
    Fate.UNDECIDED: "Blastomer (undifferenziert)",
    Fate.TE: "Trophoblast (außen)",
    Fate.ICM: "Embryoblast (innen)",
}


def _build_fate_matrix(
    default: float,
    overrides: dict[tuple[Fate, Fate], float],
) -> np.ndarray:
    """Erzeugt eine symmetrische `(N_fates, N_fates)`-Matrix aus paarweisen Vorgaben.

    Skaliert automatisch mit der Anzahl der Einträge in `Fate`, sodass spätere
    Schicksale (Epiblast/Hypoblast) ohne manuelles Umschreiben von Index-Tabellen
    ergänzt werden können.
    """
    n_fates = len(Fate)
    mat = np.full((n_fates, n_fates), default, dtype=float)
    for (f_a, f_b), val in overrides.items():
        mat[int(f_a), int(f_b)] = val
        mat[int(f_b), int(f_a)] = val
    return mat


def _default_adhesion() -> np.ndarray:
    """Standard-Adhäsionsmatrix `a[fate_i, fate_j]` (Differential-Adhesion-Hypothese).

    * UNDECIDED mit allem: neutral (1.0)
    * TE <-> TE: verstärkt (1.3) -> fördert epithelialen Zusammenhalt
    * TE <-> ICM: abgeschwächt (0.9) -> erleichtert Entmischung
    * ICM <-> ICM: stark (2.0) -> fördert kompakte innere Zellmasse
    """
    return _build_fate_matrix(
        default=1.0,
        overrides={
            (Fate.TE, Fate.TE): 1.3,
            (Fate.TE, Fate.ICM): 0.9,
            (Fate.ICM, Fate.ICM): 2.0,
        },
    )


def _default_repulsion() -> np.ndarray:
    """Standard-Abstoßungsmatrix `rep[fate_i, fate_j]` zur Skalierung von `k_rep`.

    * UNDECIDED <-> UNDECIDED: weich (0.4) -> frühe Blastomere können sich in der
      engen Zona pellucida gegeneinander abflachen
    * TE <-> TE & ICM <-> ICM: normal (1.0) -> klare Zellabgrenzung ohne Verklumpen
    * TE <-> ICM: verstärkt (1.8) -> scharfe Trennung zwischen Trophoblast und Embryoblast
    """
    return _build_fate_matrix(
        default=1.0,
        overrides={
            (Fate.UNDECIDED, Fate.UNDECIDED): 0.4,
            (Fate.TE, Fate.ICM): 1.8,
        },
    )


# --------------------------------------------------------------------------
# Reine mathematische Hilfsfunktionen (gemeinsam genutzt von Teilmodellen)
# --------------------------------------------------------------------------
def _pair_force(
    gap: np.ndarray,
    k_rep: float | np.ndarray,
    k_adh: float | np.ndarray,
    w: float | np.ndarray,
) -> np.ndarray:
    """Paarweise Radialkraft: lineare Abstoßung bei Überlappung + glatte Anziehung in (0, w).

    Positives Vorzeichen zieht Partikel aufeinander zu (in Richtung `u_ij`), negatives
    stößt sie auseinander. Die Anziehung folgt einem parabolischen Buckel `4*x*(1-x)`
    auf `x = gap / w`, der an beiden Rändern (`gap = 0` und `gap = w`) stetig verschwindet.
    """
    rep = np.where(gap < 0.0, k_rep * gap, 0.0)
    w_safe = np.maximum(w, _EPS)
    x = np.clip(gap / w_safe, 0.0, 1.0)
    adh = np.where((gap > 0.0) & (gap < w), k_adh * w * 4.0 * x * (1.0 - x), 0.0)
    return rep + adh


def _gaussian_kernel(sigma: float, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt Offsets und Gewichte eines diskreten 1D-Gauß-Kerns (`±3*sigma`)."""
    sig = max(sigma, 1e-6)
    half = int(np.ceil(3.0 * sig))
    offs = np.arange(-half, half + 1)
    weights = np.exp(-0.5 * (offs / sig) ** 2)
    if normalize:
        weights /= weights.sum()
    return offs, weights


def _convolve_periodic(arr: np.ndarray, kernel: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Periodische 1D-Faltung auf einem geschlossenen Ring mittels `np.roll`."""
    offs, weights = kernel
    out = np.zeros_like(arr, dtype=float)
    for o, w in zip(offs, weights):
        out += w * np.roll(arr, int(o))
    return out


def _angle_bins(xy: np.ndarray, n_bins: int) -> np.ndarray:
    """Ordnet 2D-Vektoren `xy` (relativ zum jeweiligen Zentrum) Winkelsektoren `[0, n_bins)` zu."""
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return ((theta + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins


# --------------------------------------------------------------------------
# Parameter-Konfiguration
# --------------------------------------------------------------------------
@dataclass
class Params:
    """Zentrale Parameter-Konfiguration für alle Teilmodelle der Simulation.

    Alle Zeitangaben sind grob in Stunden ('h') kalibriert; Längen sind in
    willkürlichen Einheiten relativ zur Zona pellucida (`zp_radius = 1.5`) angegeben.
    """

    # --- Geometrie / Mechanik -------------------------------------------
    r_zygote: float = 1.40         # Startradius der Zygote (füllt zp_radius=1.5 bis auf schmalen Spalt)
    r_min: float = 0.20            # Mindestradius: darunter keine weitere Verkleinerung
    shrink: float = 2 ** -0.5      # Radiusfaktor pro Teilung (2D: Fläche halbiert sich)
    k_rep: float = 5.0             # Basis-Abstoßungsrate bei Überlappung (1/h; skaliert mit `repulsion`)
    k_adh: float = 1.0             # Adhäsionsstärke (1/h)
    adh_range: float = 0.5         # Reichweite der Adhäsion relativ zur Kontaktdistanz
    motility: float = 0.3          # Zufallsbewegung (relativ zum Zellradius, pro sqrt(h))
    dt: float = 0.025              # Zeitschritt (h)

    # --- Zellzyklus -----------------------------------------------------
    cycle_time: float = 12.0       # Mittlere Zykluslänge (h)
    cycle_cv: float = 0.10         # Variationskoeffizient (asynchrones Teilen)
    max_cells: int = 34            # Darüber hinaus teilen sich keine Zellen mehr (Zellzyklus-Arrest)

    # --- Zellschicksal --------------------------------------------------
    compaction_n: int = 8          # Ab dieser Zellzahl werden Schicksale entschieden
    neighbor_factor: float = 1.3   # Nachbar, wenn Abstand < factor * (r_i + r_j)
    signal_tau: float = 1.0        # Glättungszeit des Expositionssignals (h)
    outer_threshold: float = 0.45  # Signal >= Schwelle -> Trophoblast
    inner_threshold: float = 0.20  # Signal <= Schwelle -> Embryoblast
    commit_time: float = 3.0       # So lange muss das Signal konstant über/unter der Schwelle liegen (h)
    lock_n: int = 28               # Ab dieser Zellzahl sind Schicksale endgültig (Lock-in)
    te_planar_division: bool = True  # Trophoblast teilt sich tangential zur Oberfläche (einschichtig)
    planar_noise: float = 0.4      # Streuung der Teilungsrichtung (rad)

    # --- Tight Junctions (Epithel-Zellverbindungen im TE-Ring nach Lock-in) -------
    tj_enabled: bool = True        # Aktiviert laterale TE-TE-Verbindungen ab dem Lock-in
    k_tj: float = 6.0              # Federsteifigkeit der Tight Junctions unter Zug (1/h)
    tj_form_range: float = 1.25    # Bindungsbildung bei Abstand < form_range * (r_i + r_j)
    tj_break_range: float = 1.80   # Bruchgrenze: Bindung reißt bei Abstand > break_range * (r_i + r_j)
    tj_max_bonds: int = 2          # Max. laterale Bindungen pro TE-Zelle (2 -> einschichtiger Ring)
    tj_lateral_cos: float = 0.15   # Max. Kosinus zwischen zwei Bindungen derselben Zelle (~81° Mindestwinkel)

    # --- Zona pellucida (starre Hülle um den Embryo, Mittelpunkt = Ursprung) -------
    zp_enabled: bool = True
    zp_radius: float = 1.5         # Innenradius der Zona (mit r_zygote=1.40 -> schmaler perivitelliner Spalt)
    zp_thickness: float = 0.15     # Dicke (Wandmitte bei zp_radius + Dicke/2)
    zp_segments: int = 72          # Auflösung des Rings (Segmente à 5 Grad)
    k_wall: float = 12.0           # Steifigkeit der starren Zona-Wand (1/h)
    zp_enzyme: float = 0.01        # Abbaurate je TE-Zelle im Wandkontakt (Festigkeit/h) -> lytische Enzyme
    zp_stress: float = 0.50        # Abbaurate je (relative Überlappung)^2 und Zelle (Festigkeit/h) -> Druck
    zp_footprint: float = 2.0      # Breite (sigma, in Segmenten) des Wandabschnitts, den eine Zelle schädigt
    zp_variability: float = 0.30   # Streuung der Anfangsfestigkeit (Materialinhomogenität)
    zp_corr_len: float = 8.0       # Korrelationslänge dieser Inhomogenität (in Segmenten)
    zp_feedback: float = 2.0       # Rückkopplung: ausgedünnte Segmente werden stärker belastet (0 = aus)

    # --- Blastozoel (Partikel-Fluid, gepumpt von TE-Zellen nach Lock-in) -----------
    lumen_enabled: bool = True
    cavity_n: int = 28             # Mindestzellzahl für Beginn des Flüssigkeits-Pumpens (zusätzlich: Lock-in)
    cavity_min_te: int = 6         # Mindestzahl Trophoblastzellen
    lumen_pump: float = 0.25       # Pumprate: erzeugte Fluidpartikel pro TE-Zelle und Stunde (0 = Pumpe aus)
    fluid_max: int = 80            # Ceiling: maximale Anzahl Fluidpartikel im Blastozoel
    fluid_hatch_factor: float = 0.25  # Multiplikator der Pumprate nach Öffnen der Zona pellucida
    fluid_r: float = 0.11          # Radius eines Fluidpartikels
    fluid_cell_scale: float = 1.45 # Faktor für Fluid-Zell-Kontaktdistanz (verhindert Durchrutschen im TE)
    k_fluid_rep: float = 10.0      # Fluid-Fluid-Abstoßung (hydrostatischer Druck, 1/h)
    k_fluid_coh: float = 1.2       # Leichte Fluid-Fluid-Kohäsion (Oberflächenspannung -> Mikrolumen verschmelzen)
    fluid_coh_range: float = 0.9   # Reichweite der Fluid-Kohäsion relativ zu `2 * fluid_r`
    k_lumen: float = 3.5           # Druckkraft Fluid -> Zellen (1/h)
    k_lumen_back: float = 8.0      # Gegenkraft Zellen -> Fluid (1/h; Fluid weicht schneller aus als Zellen)
    lumen_icm_soft: float = 0.40   # Innere Zellmasse wird weicher weggedrückt als TE (wölbt sich vor)
    fluid_motility: float = 0.25   # Thermische Beweglichkeit der Fluidpartikel (relativ zu `fluid_r`, pro sqrt(h))

    adhesion: np.ndarray = field(default_factory=_default_adhesion)
    repulsion: np.ndarray = field(default_factory=_default_repulsion)


# Vollständige Liste aller zellbezogenen Zustands-Arrays (Structure-of-Arrays).
# `_select()` und `_divide()` iterieren über dieses Tupel, damit alle Arrays synchron bleiben.
_CELL_FIELDS: tuple[str, ...] = (
    "pos",      # (N, 2) float: Zellmittelpunkte
    "r",        # (N,) float: Zellradien
    "age",      # (N,) float: Zeit seit der letzten Teilung (h)
    "cycle",    # (N,) float: individuelle Zellzykluslänge (h)
    "fate",     # (N,) int8: Zellschicksal (`Fate`)
    "signal",   # (N,) float: geglättete Exposition (0 = innen, 1 = frei)
    "lean",     # (N,) int8: momentane Tendenz (+1 -> TE, -1 -> ICM, 0 -> neutral)
    "timer",    # (N,) float: Dauer der aktuellen konstanten Tendenz (h)
    "gen",      # (N,) int: Anzahl bisheriger Teilungen
    "uid",      # (N,) int: eindeutige Zell-ID (Vorbereitung für späteres Lineage-Tracing)
    "outside",  # (N,) bool: True, wenn die Zelle die Zona durch eine Öffnung verlassen hat
)


class Simulation:
    """Agentenbasierte 2D-Simulation von Furchung, Morula, Blastozyste und Zona-Öffnung.

    Der gesamte Zellzustand ist vektorisiert in parallelen NumPy-Arrays (`_CELL_FIELDS`)
    gespeichert; `uid` wird bereits eindeutig fortgeschrieben und kann später zusammen mit
    einer `parent_uid` für vollständiges Lineage-Tracing genutzt werden.

    Ein Simulationsschritt (`step()`) koppelt:
      1. Mechanische Relaxation (schicksalsabhängige Abstoßung & Adhäsion, Motilität),
      2. Laterale Tight Junctions im TE-Ring nach dem Lock-in (Zugfedern, Bildung, Bruch),
      3. Partikel-Blastozoel (TE-Pumpen bis Ceiling, Fluiddruck, Oberflächenspannung, Leckage),
      4. Wechselwirkung mit der segmentierten Zona pellucida (Wandkraft + Abbau),
      5. Geometrische Expositionsmessung und zeitverzögerte Schicksalsentscheidung,
      6. Zellzyklus-Fortschritt und Zellteilung (inkl. Tight-Junction-Vererbung).
    """

    def __init__(self, params: Params | None = None, seed: int = 0):
        """Initialisiert die Simulation im Einzellstadium (Zygote im Ursprung)."""
        self.p = params if params is not None else Params()
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.locked = False  # Wird True, sobald `self.n >= self.p.lock_n` gilt
        self._next_uid = 0   # Monoton steigender Zähler für eindeutige Zell-IDs

        # Startzustand: eine einzelne, undifferenzierte Zygote im Ursprung (0, 0)
        self.pos = np.zeros((1, 2), dtype=float)
        self.r = np.array([self.p.r_zygote], dtype=float)
        self.age = np.zeros(1, dtype=float)
        self.cycle = np.array([self._draw_cycle()], dtype=float)
        self.fate = np.array([Fate.UNDECIDED], dtype=np.int8)
        self.signal = np.ones(1, dtype=float)  # Isolierte Zygote startet vollständig exponiert (= 1.0)
        self.lean = np.zeros(1, dtype=np.int8)
        self.timer = np.zeros(1, dtype=float)
        self.gen = np.zeros(1, dtype=int)
        self.uid = np.array([self._new_uid()], dtype=int)
        self.outside = np.zeros(1, dtype=bool)

        # Zona pellucida: Anfangsfestigkeit je Winkelsegment (1 = intakt, <= 0 = offen)
        # sowie vorab berechneter Gauß-Kern für die räumliche Verteilung des Wandabbaus
        self.zp_I = self._initial_zp()
        self._zp_kernel = _gaussian_kernel(self.p.zp_footprint, normalize=True)
        self.t_hatch: float | None = None  # Zeitpunkt der ersten Öffnung (Segment <= 0)

        # Blastozoel: (M, 2)-Array aus Positionen der Fluidpartikel
        self.fluid_pos = np.zeros((0, 2), dtype=float)
        self.t_cavity: float | None = None  # Zeitpunkt der ersten Fluid-Generierung (Kavitation)

        # Tight Junctions: (K, 2)-Array aus sortierten Zell-Index-Paaren (a < b), aktiv nach Lock-in
        self.tj_bonds = np.zeros((0, 2), dtype=int)

    def _initial_zp(self) -> np.ndarray:
        """Erzeugt das räumlich korrelierte Anfangs-Festigkeitsprofil der Zona pellucida."""
        p = self.p
        noise = self.rng.standard_normal(p.zp_segments)
        kernel = _gaussian_kernel(p.zp_corr_len, normalize=False)
        smooth = _convolve_periodic(noise, kernel)
        smooth = smooth / (smooth.std() + _EPS)
        return np.clip(1.0 + p.zp_variability * smooth, 0.4, 1.6)

    # ------------------------------------------------------------------
    # Hilfsfunktionen & Schicksals-Prädikate
    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        """Aktuelle Gesamtzahl der Zellen im Embryo."""
        return len(self.r)

    def _is_te(self) -> np.ndarray:
        """Bool-Maske aller Trophoblastzellen."""
        return self.fate == Fate.TE

    def _is_inner_mass(self) -> np.ndarray:
        """Bool-Maske der inneren Zellmasse (aktuell ICM; später z. B. EPI | HYP)."""
        return self.fate == Fate.ICM

    def _new_uid(self) -> int:
        """Vergibt eine neue, fortlaufende Zell-ID und erhöht den internen Zähler."""
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def _draw_cycle(self, size: int | None = None) -> float | np.ndarray:
        """Zieht eine oder mehrere zufällige Zellzykluslängen (nach unten abgeschnitten)."""
        p = self.p
        c = p.cycle_time * (1.0 + p.cycle_cv * self.rng.standard_normal(size))
        return np.maximum(c, 0.3 * p.cycle_time)

    def _select(self, mask: np.ndarray) -> None:
        """Filtert alle Zell-Arrays in `_CELL_FIELDS` und passt `tj_bonds` an neue Indizes an."""
        old_n = self.n
        for name in _CELL_FIELDS:
            setattr(self, name, getattr(self, name)[mask])
        if len(self.tj_bonds):
            remap = np.full(old_n, -1, dtype=int)
            remap[mask] = np.arange(self.n)
            a = remap[self.tj_bonds[:, 0]]
            b = remap[self.tj_bonds[:, 1]]
            keep = (a >= 0) & (b >= 0)
            if keep.any():
                self.tj_bonds = np.stack(
                    [np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])], axis=1
                )
            else:
                self.tj_bonds = np.zeros((0, 2), dtype=int)

    # ------------------------------------------------------------------
    # Mechanik
    # ------------------------------------------------------------------
    def _neighbor_pairs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ermittelt alle wechselwirkenden Zellpaare `(i < j)` mittels KD-Tree.

        Returns:
            `(i, j, u, d, s)` mit Indizes `i < j`, Einheitsvektor `u` von `i` nach `j`,
            Abstand `d` und Kontaktdistanz `s = r_i + r_j`.
        """
        p = self.p
        if self.n < 2:
            empty_i = np.zeros(0, dtype=int)
            empty_f = np.zeros(0, dtype=float)
            return empty_i, empty_i, np.zeros((0, 2), dtype=float), empty_f, empty_f

        max_fac = max(1.0 + p.adh_range, p.neighbor_factor, p.tj_form_range)
        reach = 2.0 * float(self.r.max()) * max_fac
        pairs = cKDTree(self.pos).query_pairs(reach, output_type="ndarray")
        if len(pairs) == 0:
            empty_i = np.zeros(0, dtype=int)
            empty_f = np.zeros(0, dtype=float)
            return empty_i, empty_i, np.zeros((0, 2), dtype=float), empty_f, empty_f

        i, j = pairs[:, 0], pairs[:, 1]
        dvec = self.pos[j] - self.pos[i]
        d = np.maximum(np.linalg.norm(dvec, axis=1), _EPS)
        s = self.r[i] + self.r[j]
        keep = d < s * max_fac
        return i[keep], j[keep], dvec[keep] / d[keep, None], d[keep], s[keep]

    def _mechanics(
        self,
        i: np.ndarray,
        j: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        s: np.ndarray,
    ) -> None:
        """Aktualisiert Zellpositionen über überdämpfte Paarkräfte und Zufallsbewegung."""
        p = self.p
        if len(i):
            gap = d - s
            w = p.adh_range * s
            k_rep = p.k_rep * p.repulsion[self.fate[i], self.fate[j]]
            k_adh = p.k_adh * p.adhesion[self.fate[i], self.fate[j]]
            move = _pair_force(gap, k_rep, k_adh, w) * p.dt
            disp = np.zeros_like(self.pos)
            np.add.at(disp, i, move[:, None] * u)
            np.add.at(disp, j, -move[:, None] * u)
            self.pos += disp

        self.pos += (
            p.motility * self.r[:, None] * np.sqrt(p.dt) * self.rng.standard_normal(self.pos.shape)
        )

    # ------------------------------------------------------------------
    # Tight Junctions (TE-Epithelverbindungen nach Lock-in)
    # ------------------------------------------------------------------
    def _tj_apply_forces(self) -> None:
        """Entfernt gerissene/ungültige Tight Junctions und wendet Zugfederkräfte an."""
        p = self.p
        if len(self.tj_bonds) == 0:
            return

        a, b = self.tj_bonds[:, 0], self.tj_bonds[:, 1]
        dvec = self.pos[b] - self.pos[a]
        dist = np.maximum(np.linalg.norm(dvec, axis=1), _EPS)
        s_b = self.r[a] + self.r[b]

        alive = (self.fate[a] == Fate.TE) & (self.fate[b] == Fate.TE) & (dist <= p.tj_break_range * s_b)
        if not alive.all():
            self.tj_bonds = self.tj_bonds[alive]
            a, b, dvec, dist, s_b = a[alive], b[alive], dvec[alive], dist[alive], s_b[alive]

        if len(a) == 0:
            return

        stretch = np.maximum(dist - s_b, 0.0)
        if stretch.any():
            u_b = dvec / dist[:, None]
            move = (p.k_tj * stretch * p.dt)[:, None] * u_b
            disp = np.zeros_like(self.pos)
            np.add.at(disp, a, move)
            np.add.at(disp, b, -move)
            self.pos += disp

    def _tj_form_bonds(
        self,
        i: np.ndarray,
        j: np.ndarray,
        d: np.ndarray,
        s: np.ndarray,
    ) -> None:
        """Knüpft neue laterale Tight Junctions zwischen nahen TE-Paaren (kürzeste zuerst)."""
        p = self.p
        if len(i) == 0:
            return

        te = self._is_te()
        cand = te[i] & te[j] & (d < p.tj_form_range * s)
        if not cand.any():
            return

        ci, cj, cd = i[cand], j[cand], d[cand]
        adj: list[list[int]] = [[] for _ in range(self.n)]
        existing: set[tuple[int, int]] = set()
        for a0, b0 in self.tj_bonds:
            ia, ib = int(a0), int(b0)
            adj[ia].append(ib)
            adj[ib].append(ia)
            existing.add((ia, ib))

        def _lateral_ok(cell: int, nb: int) -> bool:
            """Prüft Gradlimit und lateralen Mindestwinkel zu bereits bestehenden Bindungen."""
            if len(adj[cell]) >= p.tj_max_bonds:
                return False
            if not adj[cell]:
                return True
            v_new = self.pos[nb] - self.pos[cell]
            v_new = v_new / max(float(np.linalg.norm(v_new)), _EPS)
            for other in adj[cell]:
                v_old = self.pos[other] - self.pos[cell]
                v_old = v_old / max(float(np.linalg.norm(v_old)), _EPS)
                if float(np.dot(v_new, v_old)) > p.tj_lateral_cos:
                    return False
            return True

        added: list[tuple[int, int]] = []
        for idx in np.argsort(cd):
            # `cKDTree.query_pairs` garantiert bereits `u0 < v0`
            u0, v0 = int(ci[idx]), int(cj[idx])
            if (u0, v0) in existing:
                continue
            if _lateral_ok(u0, v0) and _lateral_ok(v0, u0):
                added.append((u0, v0))
                existing.add((u0, v0))
                adj[u0].append(v0)
                adj[v0].append(u0)

        if added:
            new_arr = np.array(added, dtype=int)
            self.tj_bonds = np.vstack([self.tj_bonds, new_arr]) if len(self.tj_bonds) else new_arr

    def _tj_step(
        self,
        i: np.ndarray,
        j: np.ndarray,
        d: np.ndarray,
        s: np.ndarray,
    ) -> None:
        """Verwaltet Tight Junctions im TE-Ring nach dem Lock-in."""
        if not (self.p.tj_enabled and self.locked):
            return
        self._tj_apply_forces()
        self._tj_form_bonds(i, j, d, s)

    @property
    def n_tj(self) -> int:
        """Aktuelle Anzahl aktiver Tight Junctions im TE-Ring."""
        return len(self.tj_bonds)

    # ------------------------------------------------------------------
    # Zellschicksal
    # ------------------------------------------------------------------
    def _exposure(
        self,
        i: np.ndarray,
        j: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        s: np.ndarray,
    ) -> np.ndarray:
        """Berechnet die geometrische Oberflächenexposition jeder Zelle im Intervall `[0, 1]`.

        Für jede Zelle werden die Einheitsvektoren zu ihren direkten Nachbarn gemittelt:
        * Im Embryo-Inneren umgeben Nachbarn die Zelle von allen Seiten -> die Vektoren
          heben sich weitgehend auf (`||sum(u_ij)|| / cnt ~ 0`).
        * Am Rand zeigen alle Nachbarvektoren gebündelt nach innen -> großer Betrag (`~ 1`).
        """
        close = d < self.p.neighbor_factor * s
        ic, jc, uc = i[close], j[close], u[close]

        vec = np.zeros_like(self.pos)
        cnt = np.zeros(self.n, dtype=float)
        np.add.at(vec, ic, uc)
        np.add.at(vec, jc, -uc)
        np.add.at(cnt, ic, 1.0)
        np.add.at(cnt, jc, 1.0)

        exp = np.ones(self.n, dtype=float)
        has = cnt > 0
        exp[has] = np.linalg.norm(vec[has], axis=1) / cnt[has]
        return exp

    def _update_fate(self, exposure: np.ndarray) -> None:
        """Glättet das Expositionssignal (Tiefpass) und weist Schicksale zeitverzögert zu."""
        p = self.p
        self.signal += (exposure - self.signal) * (p.dt / p.signal_tau)

        if self.n < p.compaction_n:
            return
        if self.n >= p.lock_n:
            self.locked = True

        lean = np.zeros(self.n, dtype=np.int8)
        lean[self.signal >= p.outer_threshold] = 1
        lean[self.signal <= p.inner_threshold] = -1

        same = (lean == self.lean) & (lean != 0)
        self.timer = np.where(same, self.timer + p.dt, 0.0)
        self.lean = lean

        ready = self.timer >= p.commit_time
        if self.locked:
            # Nach Lock-in dürfen nur noch bislang undifferenzierte Zellen ein Schicksal annehmen
            ready &= self.fate == Fate.UNDECIDED

        self.fate[ready & (lean == 1)] = Fate.TE
        self.fate[ready & (lean == -1)] = Fate.ICM

    # ------------------------------------------------------------------
    # Zona pellucida
    # ------------------------------------------------------------------
    def _zp_step(self) -> None:
        """Führt Wand-Mechanik und lokalen Abbau der Zona pellucida aus."""
        p = self.p
        if not p.zp_enabled:
            return

        nseg = p.zp_segments
        rho = np.linalg.norm(self.pos, axis=1)
        seg = _angle_bins(self.pos, nseg)

        closed = self.zp_I > 0.0
        half_t = 0.5 * p.zp_thickness
        r_mid = p.zp_radius + half_t

        # Richtungs-Latch: `outside` wird ausschließlich in offenen Sektoren (`in_gap`)
        # aktualisiert. So prallen bereits geschlüpfte Zellen anschließend an geschlossenen
        # Nachbarsegmenten nach außen ab, statt wieder nach innen zurückgezogen zu werden.
        in_gap = ~closed[seg]
        self.outside[in_gap] = rho[in_gap] > r_mid

        dist = np.where(self.outside, rho - r_mid, r_mid - rho)
        reach = self.r + half_t
        overlap = np.where(closed[seg], np.maximum(reach - dist, 0.0), 0.0)
        if overlap.any():
            sign = np.where(self.outside, 1.0, -1.0)
            radial = self.pos / np.maximum(rho, _EPS)[:, None]
            self.pos += (sign * p.k_wall * p.dt * overlap)[:, None] * radial

        # Lokaler Abbau durch Innendruck (quadratisch in relativer Überlappung) und TE-Enzyme
        inside = ~self.outside
        rel = overlap / self.r
        stress = np.bincount(seg, weights=np.where(inside, rel ** 2, 0.0), minlength=nseg)
        touch = inside & closed[seg] & (dist < reach + _ZP_ENZYME_MARGIN) & self._is_te()
        enzyme = np.bincount(seg[touch], minlength=nseg).astype(float)
        rate = _convolve_periodic(p.zp_enzyme * enzyme + p.zp_stress * stress, self._zp_kernel)

        # Positive Rückkopplung: ausgedünnte Segmente werden überproportional weiter abgebaut
        rate *= np.clip(self.zp_I, 0.2, None) ** (-p.zp_feedback)
        self.zp_I = np.where(closed, self.zp_I - rate * p.dt, self.zp_I)

        if self.t_hatch is None and (self.zp_I <= 0.0).any():
            self.t_hatch = self.t

    @property
    def hatched(self) -> bool:
        """True, sobald mindestens ein Segment der Zona pellucida geöffnet wurde."""
        return self.t_hatch is not None

    @property
    def n_outside(self) -> int:
        """Anzahl der Zellen, die sich aktuell außerhalb der Zona pellucida befinden."""
        return int(self.outside.sum())

    # ------------------------------------------------------------------
    # Blastozoel (Partikel-Fluid)
    # ------------------------------------------------------------------
    @property
    def n_fluid(self) -> int:
        """Aktuelle Anzahl der Blastozoel-Fluidpartikel."""
        return len(self.fluid_pos)

    @property
    def has_lumen(self) -> bool:
        """True, sobald mindestens ein Fluidpartikel im Blastozoel vorhanden ist."""
        return len(self.fluid_pos) > 0

    @property
    def lumen_pos(self) -> np.ndarray:
        """Schwerpunkt der aktuellen Blastozoel-Fluidmasse (oder Ursprung, falls leer)."""
        return self.fluid_pos.mean(axis=0) if len(self.fluid_pos) else np.zeros(2, dtype=float)

    @property
    def lumen_area(self) -> float:
        """Effektive 2D-Fläche des Partikel-Blastozoels (inkl. 2D-Packungsfaktor)."""
        return float(len(self.fluid_pos) * np.pi * (self.p.fluid_r ** 2) * _LUMEN_PACKING_FACTOR)

    @property
    def lumen_radius(self) -> float:
        """Äquivalenter Kreisradius der aktuellen Blastozoel-Fläche (0.0 ohne Fluid)."""
        return float(np.sqrt(self.lumen_area / np.pi)) if self.has_lumen else 0.0

    def _pump_fluid(self) -> None:
        """Erzeugt neue Fluidpartikel basolateral an Trophoblastzellen nach dem Lock-in."""
        p = self.p
        te_mask = self._is_te()
        if not (self.locked and self.n >= p.cavity_n and int(te_mask.sum()) >= p.cavity_min_te):
            return
        if p.lumen_pump <= 0.0 or len(self.fluid_pos) >= p.fluid_max:
            return

        rate = p.lumen_pump * (p.fluid_hatch_factor if self.hatched else 1.0)
        prob = min(1.0, rate * p.dt)
        te_idx = np.flatnonzero(te_mask)
        spawning = te_idx[self.rng.random(len(te_idx)) < prob]
        room = p.fluid_max - len(self.fluid_pos)
        if len(spawning) > room:
            spawning = self.rng.choice(spawning, room, replace=False)
        if len(spawning) == 0:
            return

        center = self.pos.mean(axis=0)
        inner_mask = self._is_inner_mass()
        has_inner = bool(inner_mask.any())
        inner_pos = self.pos[inner_mask]
        inner_r = self.r[inner_mask]

        new_pts: list[np.ndarray] = []
        for ci in spawning:
            inward = center - self.pos[ci]
            norm = float(np.linalg.norm(inward))
            if norm < 1e-6:
                ang = self.rng.uniform(0.0, 2.0 * np.pi)
                inward = np.array([np.cos(ang), np.sin(ang)])
            else:
                inward = inward / norm
            cand_pos = self.pos[ci] + inward * (self.r[ci] + p.fluid_r * _FLUID_SPAWN_OFFSET)

            # Polares TE mit direktem Kontakt zur inneren Zellmasse pumpt nicht dort hinein
            if has_inner:
                d_inner = np.linalg.norm(inner_pos - cand_pos, axis=1)
                if (d_inner < inner_r * _ICM_PUMP_BLOCK_FACTOR).any():
                    continue
            new_pts.append(cand_pos)

        if new_pts:
            arr = np.array(new_pts, dtype=float)
            self.fluid_pos = np.vstack([self.fluid_pos, arr]) if len(self.fluid_pos) else arr
            if self.t_cavity is None:
                self.t_cavity = self.t

    def _remove_leaked_fluid(self) -> None:
        """Entfernt Fluidpartikel, die durch Lücken im TE-Ring nach außen ausgetreten sind.

        Geometrische Annahme: Die Winkel-Sektoren werden um den Zellschwerpunkt (`pos.mean`)
        aufgespannt. Bei stark asymmetrischer Blastozyste oder späterer Implantation kann der
        Schwerpunkt näher am embryonalen Pol liegen; für geschlossene bis schlüpfende
        Blastozysten reicht die Sektor-Hüllkurve aber zuverlässig aus.
        """
        center = self.pos.mean(axis=0)
        rel_c = self.pos - center
        rel_f = self.fluid_pos - center

        rho_c = np.linalg.norm(rel_c, axis=1) + self.r
        rho_f = np.linalg.norm(rel_f, axis=1)

        sec_c = _angle_bins(rel_c, _LEAKAGE_SECTORS)
        sec_f = _angle_bins(rel_f, _LEAKAGE_SECTORS)

        sec_max = np.zeros(_LEAKAGE_SECTORS, dtype=float)
        np.maximum.at(sec_max, sec_c, rho_c)
        keep = rho_f <= sec_max[sec_f]
        if not keep.all():
            self.fluid_pos = self.fluid_pos[keep]

    def _lumen_step(self) -> None:
        """Steuert das Partikel-Blastozoel: TE-Pumpen, Fluiddynamik und Leckage.

        Ablauf pro Zeitschritt:
          1. **Pumpen**: Trophoblastzellen erzeugen nach Lock-in Fluidpartikel bis `fluid_max`.
          2. **Fluid-Fluid-Mechanik**: Kurzreichweitige Abstoßung (`k_fluid_rep`) erzeugt
             isotropen hydrostatischen Druck; schwache Kohäsion (`k_fluid_coh`) wirkt als
             Oberflächenspannung und verschmilzt Mikrolumen zu einer zusammenhängenden Höhle.
          3. **Fluid-Zell-Mechanik**: Fluidpartikel drücken Zellen nach außen (`k_lumen`,
             innere Zellmasse weicher um `lumen_icm_soft`), während Zellen das Fluid stärker
             zurückdrücken (`k_lumen_back`), sodass es flexibel um das ICM herumfließt.
          4. **Leckage**: Partikel außerhalb der lokalen Embryo-Hüllkurve laufen aus.
        """
        p = self.p
        if not p.lumen_enabled:
            return

        self._pump_fluid()
        m = len(self.fluid_pos)
        if m == 0:
            return

        # Fluid <-> Fluid Mechanik (voll vektorisiert, ideal für kleines M <= fluid_max)
        s_ff = 2.0 * p.fluid_r
        w_coh = p.fluid_coh_range * s_ff
        if m >= 2:
            dvec = self.fluid_pos[None, :, :] - self.fluid_pos[:, None, :]  # (M, M, 2)
            d = np.linalg.norm(dvec, axis=2)
            np.fill_diagonal(d, np.inf)
            force = _pair_force(d - s_ff, p.k_fluid_rep, p.k_fluid_coh, w_coh)
            move = force * p.dt / np.maximum(d, _EPS)
            self.fluid_pos += (move[:, :, None] * dvec).sum(axis=1)

        # Fluid <-> Zelle Mechanik (voll vektorisiert: (N, M, 2))
        dvec_cf = self.pos[:, None, :] - self.fluid_pos[None, :, :]
        d_cf = np.maximum(np.linalg.norm(dvec_cf, axis=2), _EPS)
        s_fc = self.r[:, None] + p.fluid_cell_scale * p.fluid_r
        ov = np.maximum(s_fc - d_cf, 0.0)
        if ov.any():
            u_cf = dvec_cf / d_cf[:, :, None]
            stiff = np.where(self._is_inner_mass(), p.lumen_icm_soft, 1.0)[:, None]
            self.pos += ((p.k_lumen * p.dt * stiff * ov)[:, :, None] * u_cf).sum(axis=1)
            self.fluid_pos -= ((p.k_lumen_back * p.dt * ov)[:, :, None] * u_cf).sum(axis=0)

        # Thermische Beweglichkeit der Flüssigkeit
        self.fluid_pos += (
            p.fluid_motility * p.fluid_r * np.sqrt(p.dt) * self.rng.standard_normal(self.fluid_pos.shape)
        )

        self._remove_leaked_fluid()

    def collapse_lumen(self, fraction: float = 1.0) -> None:
        """Blastozoel-Kollaps: entfernt zufällig den angegebenen Anteil der Fluidpartikel."""
        m = len(self.fluid_pos)
        if m == 0:
            return
        n_rm = int(round(float(np.clip(fraction, 0.0, 1.0)) * m))
        if n_rm >= m:
            self.fluid_pos = np.zeros((0, 2), dtype=float)
        elif n_rm > 0:
            keep = np.ones(m, dtype=bool)
            keep[self.rng.choice(m, n_rm, replace=False)] = False
            self.fluid_pos = self.fluid_pos[keep]

    # ------------------------------------------------------------------
    # Zellteilung & TJ-Vererbung
    # ------------------------------------------------------------------
    def _inherit_tj_bonds(
        self,
        ready: np.ndarray,
        old_n: int,
        old_pos: np.ndarray,
        axis: np.ndarray,
    ) -> None:
        """Vererbt bestehende Tight Junctions sauber auf Mutter- und Tochterzelle.

        Verwendet konsequent `old_pos` (Positionen *vor* dem Verschieben aller sich
        teilenden Zellen), sodass die Geometrie auch dann eindeutig und reihenfolge-
        unabhängig bleibt, wenn zwei gebundene TE-Zellen im selben Zeitschritt teilen.
        """
        p = self.p
        if not (p.tj_enabled and self.locked):
            return

        # Ordnet neu angehängte Tochterzellen der Position ihrer Mutter vor der Teilung zu
        orig_idx = np.arange(self.n, dtype=int)
        orig_idx[old_n:] = ready

        new_bonds: list[tuple[int, int]] = []
        for idx_k, parent in enumerate(ready):
            parent_i = int(parent)
            if self.fate[parent_i] != Fate.TE:
                continue
            child_i = old_n + idx_k

            if len(self.tj_bonds):
                incident = [
                    bi for bi in range(len(self.tj_bonds))
                    if self.tj_bonds[bi, 0] == parent_i or self.tj_bonds[bi, 1] == parent_i
                ]
                ax = axis[idx_k]
                p_orig = old_pos[parent_i]

                if len(incident) == 2 and p.tj_max_bonds == 2:
                    # Mutter liegt bei `p_orig + offset`, Tochter bei `p_orig - offset`:
                    # Der Nachbar mit kleinerer Projektion auf `ax` gehört zur Tochter.
                    projs = []
                    for bi in incident:
                        other = int(
                            self.tj_bonds[bi, 1] if self.tj_bonds[bi, 0] == parent_i else self.tj_bonds[bi, 0]
                        )
                        proj = float(np.dot(old_pos[orig_idx[other]] - p_orig, ax))
                        projs.append((proj, bi, other))
                    projs.sort(key=lambda item: item[0])
                    _, bi_c, other_c = projs[0]
                    self.tj_bonds[bi_c] = (min(child_i, other_c), max(child_i, other_c))
                else:
                    for bi in incident:
                        other = int(
                            self.tj_bonds[bi, 1] if self.tj_bonds[bi, 0] == parent_i else self.tj_bonds[bi, 0]
                        )
                        # Negative Projektion bedeutet: Nachbar liegt auf der Tochter-Seite (`-ax`)
                        if float(np.dot(old_pos[orig_idx[other]] - p_orig, ax)) < 0.0:
                            self.tj_bonds[bi] = (min(child_i, other), max(child_i, other))

            new_bonds.append((parent_i, child_i))

        if new_bonds:
            arr = np.array(new_bonds, dtype=int)
            self.tj_bonds = np.vstack([self.tj_bonds, arr]) if len(self.tj_bonds) else arr

    def _divide(self) -> None:
        """Teilt alle reifen Zellen (`age >= cycle`) bis zur Obergrenze `max_cells`."""
        p = self.p
        ready = np.flatnonzero(self.age >= self.cycle)
        free = p.max_cells - self.n
        if len(ready) == 0 or free <= 0:
            return
        if len(ready) > free:
            ready = self.rng.choice(ready, free, replace=False)

        k = len(ready)
        old_n = self.n
        old_pos = self.pos.copy()

        ang = self.rng.uniform(0.0, 2.0 * np.pi, k)
        if p.te_planar_division:
            te = self.fate[ready] == Fate.TE
            if te.any():
                ref = self.lumen_pos if self.has_lumen else old_pos.mean(axis=0)
                radial = old_pos[ready][te] - ref
                tangent = np.arctan2(radial[:, 1], radial[:, 0]) + 0.5 * np.pi
                ang[te] = tangent + p.planar_noise * self.rng.standard_normal(int(te.sum()))

        axis = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        r_new = np.maximum(self.r[ready] * p.shrink, p.r_min)
        offset = (r_new * _DIV_OFFSET_FACTOR)[:, None] * axis

        parent_pos = old_pos[ready]
        self.pos[ready] = parent_pos + offset
        self.r[ready] = r_new
        self.age[ready] = 0.0
        self.cycle[ready] = self._draw_cycle(k)
        self.gen[ready] += 1

        new = {
            "pos": parent_pos - offset,
            "r": r_new,
            "age": np.zeros(k, dtype=float),
            "cycle": self._draw_cycle(k),
            "fate": self.fate[ready].copy(),
            "signal": self.signal[ready].copy(),
            "lean": self.lean[ready].copy(),
            "timer": self.timer[ready].copy(),
            "gen": self.gen[ready].copy(),
            "uid": np.array([self._new_uid() for _ in range(k)], dtype=int),
            "outside": self.outside[ready].copy(),
        }
        for name in _CELL_FIELDS:
            setattr(self, name, np.concatenate([getattr(self, name), new[name]]))

        self._inherit_tj_bonds(ready, old_n, old_pos, axis)

    # ------------------------------------------------------------------
    # Zeitschritt & Ablaufsteuerung
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Führt genau einen Integrationszeitschritt `dt` der Simulation aus.

        Die Nachbarschaftsliste `(i, j, u, d, s)` wird bewusst einmal pro Zeitschritt vor
        `_mechanics()` aufgebaut und für `_tj_step()` sowie `_exposure()` wiederverwendet
        (Explizit-Euler-Näherung, vermeidet mehrfaches KD-Tree-Bauen pro Schritt).
        """
        i, j, u, d, s = self._neighbor_pairs()
        self._mechanics(i, j, u, d, s)
        self._tj_step(i, j, d, s)
        self._lumen_step()
        self._zp_step()
        self._update_fate(self._exposure(i, j, u, d, s))
        self.age += self.p.dt
        self._divide()
        self.t += self.p.dt

    def run(self, hours: float) -> Simulation:
        """Lässt die Simulation um eine feste Zeitspanne (`hours`) weiterlaufen."""
        for _ in range(int(round(hours / self.p.dt))):
            self.step()
        return self

    def run_until_n(self, n_target: int, max_hours: float = 500.0) -> Simulation:
        """Läuft weiter, bis mindestens `n_target` Zellen existieren oder `max_hours` um sind."""
        t_end = self.t + max_hours
        while self.n < n_target and self.t < t_end:
            self.step()
        return self

    # ------------------------------------------------------------------
    # Perturbationen (virtuelle Experimente)
    # ------------------------------------------------------------------
    def remove_cells(self, mask: np.ndarray) -> int:
        """Entfernt alle Zellen, bei denen `mask` True ist (mindestens eine Zelle bleibt)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.all():
            mask = mask.copy()
            mask[0] = False
        self._select(~mask)
        return int(mask.sum())

    def remove_random(self, k: int) -> int:
        """Entfernt `k` zufällig gewählte Zellen (lässt mindestens eine Zelle übrig)."""
        k = int(min(k, self.n - 1))
        if k <= 0:
            return 0
        mask = np.zeros(self.n, dtype=bool)
        mask[self.rng.choice(self.n, k, replace=False)] = True
        return self.remove_cells(mask)

    def remove_half(self, direction: tuple[float, float] = (1.0, 0.0)) -> int:
        """Entfernt alle Zellen auf der positiven Seite einer Trenngeraden durch den Schwerpunkt."""
        dvec = np.asarray(direction, dtype=float)
        dvec = dvec / np.linalg.norm(dvec)
        proj = (self.pos - self.pos.mean(axis=0)) @ dvec
        return self.remove_cells(proj > 0.0)

    def remove_in_disk(self, center: tuple[float, float], radius: float) -> int:
        """Entfernt alle Zellen, deren Mittelpunkt innerhalb einer Kreisscheibe liegt."""
        dist = np.linalg.norm(self.pos - np.asarray(center, dtype=float), axis=1)
        return self.remove_cells(dist < radius)

    def drill_zp(self, angle_deg: float = 0.0, width_deg: float = 20.0) -> None:
        """Erzeugt eine künstliche Öffnung in der Zona pellucida ('Assisted Hatching')."""
        nseg = self.p.zp_segments
        centers = (np.arange(nseg) + 0.5) / nseg * 360.0 - 180.0
        diff = np.abs((centers - angle_deg + 180.0) % 360.0 - 180.0)
        self.zp_I[diff <= 0.5 * width_deg] = 0.0
        if self.t_hatch is None and (self.zp_I <= 0.0).any():
            self.t_hatch = self.t

    def harden_zp(self, factor: float = 2.0) -> None:
        """Erhöht die Festigkeit aller noch intakten Zona-Segmente ('Zona-Härtung')."""
        self.zp_I = np.where(self.zp_I > 0.0, self.zp_I * factor, self.zp_I)

    def set_fate(self, mask: np.ndarray, fate: Fate | int) -> None:
        """Überschreibt das Zellschicksal für eine Auswahl von Zellen (z. B. Knockout-Experiment)."""
        self.fate[np.asarray(mask, dtype=bool)] = int(fate)

    # ------------------------------------------------------------------
    # Auswertung
    # ------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        """Zählt die aktuelle Zellzahl je Schicksal."""
        return {name: int((self.fate == f).sum()) for f, name in FATE_NAMES.items()}

    def summary(self) -> str:
        """Erzeugt eine kompakte einzeilige Statuszusammenfassung für Logs und Konsole."""
        c = self.counts()
        parts = [
            f"t = {self.t:5.1f} h",
            f"N = {self.n:3d}",
            (
                f"undiff. {c[FATE_NAMES[Fate.UNDECIDED]]}, "
                f"TE {c[FATE_NAMES[Fate.TE]]}, "
                f"ICM {c[FATE_NAMES[Fate.ICM]]}"
            ),
        ]
        if self.p.tj_enabled and self.locked:
            parts.append(f"TJ: {self.n_tj}")
        if self.has_lumen:
            parts.append(f"Blastozoel: R={self.lumen_radius:.2f} ({self.n_fluid} P.)")
        if self.p.zp_enabled:
            if self.hatched:
                parts.append(f"Zona: offen seit {self.t_hatch:.0f} h, {self.n_outside} Zellen draußen")
            else:
                parts.append("Zona: intakt")
        return " | ".join(parts)
