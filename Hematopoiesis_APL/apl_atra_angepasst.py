import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# ==============================================================================
# 1. ECHTZEIT-SKALIERUNG & KLINISCHE ZEITACHSE (1 Tick = 3 Stunden -> 8 Ticks/Tag)
# ==============================================================================
TICKS_PER_DAY = 8              # 1 Tick = 3 Stunden
TOTAL_DAYS = 120               # Gesamtdauer: 120 Tage (4 Monate)
MAX_TICKS = TOTAL_DAYS * TICKS_PER_DAY  # 960 Ticks

DAY_MUTATION = 25              # Tag 25: t(15;17) / PML-RARα-Mutation
DAY_ATRA_START = 55            # Tag 55: Therapiebeginn mit ATRA (nach 30 Tagen Progression)

TICK_MUTATION = DAY_MUTATION * TICKS_PER_DAY
TICK_ATRA = DAY_ATRA_START * TICKS_PER_DAY

GRID_H, GRID_W = 40, 50        # 2.000 Plätze im Knochenmark
INITIAL_HSCS = 20

# Die 5 klassischen Reifungsstufen der Myelopoese
EMPTY = 0
HSC = 1                        # 1. Hämatopoetische Stammzelle
GMP = 2                        # 2. Granulozyten-Monozyten-Progenitor
PROMYELO = 3                   # 3. Promyelozyt (APL-Blockade-Stufe)
MYELO = 4                      # 4. Myelozyt / Metamyelozyt
GRANULO = 5                    # 5. Reifer neutrophiler Granulozyt

# Realistische ATRA-Kinetik (auf 3-Stunden-Ticks kalibriert)
BASE_ATRA_CLOCK_SPEED = 0.018
POINT_OF_NO_RETURN = 0.45      # Mitose-Stopp nach transkriptionellem Umbau


# ==============================================================================
# 2. ZELL-AGENT MIT ASYNCHRONER NISCHEN- & G0-DYNAMIK
# ==============================================================================
class RealisticMyeloidCell:
    def __init__(self, stage, x, y, is_leukemic=False):
        self.stage = stage
        self.x = x
        self.y = y
        self.age = 0
        self.divisions = 0
        self.is_leukemic = is_leukemic
        self.maturation_clock = 0.0

        # Biologische Heterogenität in vivo:
        # ~70% aktive Blasten (normale Sensitivität 0.65 - 1.25)
        # ~30% ruhende G0- / tief sitzende Nischen-Blasten (reduzierte Sensitivität 0.18 - 0.40)
        self.is_quiescent = (np.random.rand() < 0.30)
        if self.is_quiescent:
            self.atra_sensitivity = np.random.uniform(0.18, 0.40)
        else:
            self.atra_sensitivity = np.random.uniform(0.65, 1.25)

    def step(self, grid, atra_active, days_on_atra):
        self.age += 1

        # 1. HSC (Stammzelle) - langsamer Umsatz, langlebig
        if self.stage == HSC:
            if np.random.rand() < 0.002:
                grid[self.x, self.y] = None
                return
            if self.age >= 10 and np.random.rand() < 0.12:
                free = get_free_neighbors(grid, self.x, self.y, bias_right=True)
                if free:
                    nx, ny = free[np.random.randint(len(free))]
                    grid[nx, ny] = RealisticMyeloidCell(GMP if np.random.rand() < 0.90 else HSC, nx, ny)
                    self.age = 0

        # 2. GMP (Progenitor) - Amplifikation (Zellzyklus ~18-24h = 6-8 Ticks)
        elif self.stage == GMP:
            if np.random.rand() < 0.01:
                grid[self.x, self.y] = None
                return
            if self.divisions >= 4 or (self.age >= 8 and np.random.rand() < 0.22):
                self.stage = PROMYELO
                self.age = 0
                return
            if self.age >= 6 and np.random.rand() < 0.32:
                free = get_free_neighbors(grid, self.x, self.y, bias_right=True)
                if free:
                    nx, ny = free[np.random.randint(len(free))]
                    child = RealisticMyeloidCell(GMP, nx, ny)
                    child.divisions = self.divisions + 1
                    grid[nx, ny] = child
                    self.divisions += 1
                    self.age = 0

        # 3. PROMYELOZYTE (Gesund vs. Leukämisch mit realistischer ATRA-Kinetik)
        elif self.stage == PROMYELO:
            if self.is_leukemic:
                # Stark verlängerte Lebensdauer (Apoptoseresistenz durch PML-RARα)
                if np.random.rand() < 0.004:
                    grid[self.x, self.y] = None
                    return

                # WIRKUNG VON ATRA:
                # 1. Pharmakokinetischer Gewebeaufbau in den ersten 3 Tagen (Ramp-up)
                # 2. Ruhende G0-Blasten werden über die Wochen nach und nach rekrutiert
                if atra_active:
                    tissue_bioavailability = min(1.0, 0.35 + 0.65 * (days_on_atra / 3.0))

                    if self.is_quiescent and np.random.rand() < 0.012:
                        self.is_quiescent = False
                        self.atra_sensitivity = np.random.uniform(0.65, 1.15)

                    self.maturation_clock += BASE_ATRA_CLOCK_SPEED * self.atra_sensitivity * tissue_bioavailability

                    if self.maturation_clock >= 1.0:
                        self.stage = MYELO
                        self.is_leukemic = False
                        self.age = 0
                        return

                # Proliferation bis zum Point of No Return (0.45)
                can_divide = (not atra_active) or (self.maturation_clock < POINT_OF_NO_RETURN)
                div_prob = 0.14 if self.is_quiescent else 0.28

                if can_divide and self.age >= 7 and np.random.rand() < div_prob:
                    free = get_free_neighbors(grid, self.x, self.y, bias_right=False)
                    if free:
                        nx, ny = free[np.random.randint(len(free))]
                        grid[nx, ny] = RealisticMyeloidCell(PROMYELO, nx, ny, is_leukemic=True)
                        self.age = 0

            else:
                # Gesunder Promyelozyt (reif in ~36-48h = 12-16 Ticks)
                if np.random.rand() < 0.01:
                    grid[self.x, self.y] = None
                    return
                self.maturation_clock += 0.085
                if self.maturation_clock >= 1.0:
                    self.stage = MYELO
                    self.age = 0
                    return
                if self.maturation_clock < POINT_OF_NO_RETURN and self.divisions < 5 and self.age >= 7 and np.random.rand() < 0.18:
                    free = get_free_neighbors(grid, self.x, self.y, bias_right=True)
                    if free:
                        nx, ny = free[np.random.randint(len(free))]
                        child = RealisticMyeloidCell(PROMYELO, nx, ny)
                        child.divisions = self.divisions + 1
                        child.maturation_clock = self.maturation_clock * 0.5
                        grid[nx, ny] = child
                        self.divisions += 1
                        self.age = 0

        # 4. MYELOZYT / METAMYELOZYT (postmitotische Reifung ~2-3 Tage)
        elif self.stage == MYELO:
            if np.random.rand() < 0.008:
                grid[self.x, self.y] = None
                return
            if self.age >= 12 and np.random.rand() < 0.18:
                self.stage = GRANULO
                self.age = 0

        # 5. REIFER GRANULOZYT (Knochenmark-Speicher & Blut: ~2.5-3.5 Tage)
        elif self.stage == GRANULO:
            if self.age >= 20 and np.random.rand() < 0.22:
                grid[self.x, self.y] = None


def get_free_neighbors(grid, x, y, bias_right=False):
    free_spots = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_H and 0 <= ny < GRID_W and grid[nx, ny] is None:
                free_spots.append((nx, ny))
                if bias_right and dy == 1:
                    free_spots.append((nx, ny))
    return free_spots


# ==============================================================================
# 3. INITIALISIERUNG & ECHTZEIT-VISUALISIERUNG (X-ACHSE IN TAGEN)
# ==============================================================================
np.random.seed(42)
grid = np.empty((GRID_H, GRID_W), dtype=object)

for _ in range(INITIAL_HSCS):
    while True:
        rx, ry = np.random.randint(0, GRID_H), np.random.randint(0, 8)
        if grid[rx, ry] is None:
            grid[rx, ry] = RealisticMyeloidCell(HSC, rx, ry)
            break

history = {"day": [], HSC: [], GMP: [], PROMYELO: [], MYELO: [], GRANULO: []}

fig, (ax_grid, ax_bars, ax_line) = plt.subplots(
    1, 3, figsize=(16.5, 5.6), gridspec_kw={"width_ratios": [1.05, 0.95, 1.45]}
)
plt.subplots_adjust(top=0.84, bottom=0.14, left=0.04, right=0.98, wspace=0.22)

stage_colors = ["#f4f4f4", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#00b4d8"]
cmap = ListedColormap(stage_colors)

img = ax_grid.imshow(np.zeros((GRID_H, GRID_W)), cmap=cmap, vmin=0, vmax=5)
ax_grid.set_xticks([])
ax_grid.set_yticks([])
ax_grid.set_xlabel("Endost (Stammzellen)  →  Sinusoide (Blut)", fontsize=8.5, fontweight="bold")
title_grid = ax_grid.set_title("Knochenmark (Tag 0)", fontweight="bold")

stage_names = ["1. HSC", "2. GMP", "3. Promyelo", "4. Myelo", "5. Granulo"]
bar_colors = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#00b4d8"]
bars = ax_bars.bar(stage_names, [0, 0, 0, 0, 0], color=bar_colors, edgecolor="black", alpha=0.88)
ax_bars.set_ylim(0, 1150)
ax_bars.set_ylabel("Zellzahl im Mark")
ax_bars.set_title("5 Reifungsstufen", fontweight="bold")
ax_bars.tick_params(axis="x", labelsize=8.5)
ax_bars.grid(axis="y", alpha=0.25)

l_prog, = ax_line.plot([], [], label="Vorläufer (HSC + GMP)", color="#33a02c", lw=1.8)
l_prom, = ax_line.plot([], [], label="Promyelozyten (Stufe 3)", color="#e31a1c", lw=2.5)
l_mat, = ax_line.plot([], [], label="Reifende & Granulozyten (4+5)", color="#00b4d8", lw=2.2)

ax_line.axvline(DAY_MUTATION, color="black", ls=":", lw=1.4)
ax_line.text(DAY_MUTATION + 1.5, 1140, f"PML-RARα\n(Tag {DAY_MUTATION})", fontsize=8, fontweight="bold")

ax_line.axvline(DAY_ATRA_START, color="darkorange", ls="--", lw=1.6)
ax_line.text(DAY_ATRA_START + 1.5, 1140, f"ATRA-Start\n(Tag {DAY_ATRA_START})", fontsize=8, color="darkorange", fontweight="bold")

# Markierung des klinischen Remissionsfensters (Tag 28-42 nach ATRA-Start)
ax_line.axvspan(DAY_ATRA_START + 28, DAY_ATRA_START + 42, color="#b7e4c7", alpha=0.35, label="Klinische Remission (Tag 28–42)")

fig.suptitle(
    "Realistische Zeitskala (1 Tick = 3h): 4- bis 5-wöchige Remissionsinduktion unter ATRA",
    fontsize=13, fontweight="bold"
)
ax_line.set_xlim(0, TOTAL_DAYS)
ax_line.set_ylim(0, 1280)
ax_line.set_xlabel("Klinischer Zeitverlauf (Tage)")
ax_line.set_ylabel("Zellzahl")
ax_line.legend(loc="upper left", fontsize=8)
ax_line.grid(alpha=0.25)


# ==============================================================================
# 4. ANIMATIONSSCHLEIFE (2 Ticks pro Frame)
# ==============================================================================
def update(frame):
    ticks_per_frame = 2
    start_tick = frame * ticks_per_frame

    for t in range(start_tick, min(start_tick + ticks_per_frame, MAX_TICKS)):
        current_day = t / TICKS_PER_DAY
        atra_active = (t >= TICK_ATRA)
        days_on_atra = max(0.0, current_day - DAY_ATRA_START)

        if t == TICK_MUTATION:
            candidates = [(x, y) for x in range(GRID_H) for y in range(GRID_W)
                          if grid[x, y] is not None and grid[x, y].stage in [GMP, PROMYELO]]
            if len(candidates) >= 4:
                for idx in np.random.choice(len(candidates), 4, replace=False):
                    cx, cy = candidates[idx]
                    grid[cx, cy] = RealisticMyeloidCell(PROMYELO, cx, cy, is_leukemic=True)

        agents = [grid[x, y] for x in range(GRID_H) for y in range(GRID_W) if grid[x, y] is not None]
        np.random.shuffle(agents)
        for agent in agents:
            if grid[agent.x, agent.y] is agent:
                agent.step(grid, atra_active, days_on_atra)

        visual_grid = np.zeros((GRID_H, GRID_W), dtype=int)
        counts = {HSC: 0, GMP: 0, PROMYELO: 0, MYELO: 0, GRANULO: 0}

        for x in range(GRID_H):
            for y in range(GRID_W):
                c = grid[x, y]
                if c is not None:
                    visual_grid[x, y] = c.stage
                    counts[c.stage] += 1

        history["day"].append(current_day)
        for k in counts:
            history[k].append(counts[k])

    day_now = history["day"][-1]
    img.set_data(visual_grid)

    if day_now < DAY_MUTATION:
        phase = "Gesunde Homöostase"
    elif day_now < DAY_ATRA_START:
        phase = f"APL-Progression (Tag {int(day_now - DAY_MUTATION)})"
    else:
        phase = f"ATRA-Therapie (Tag {int(day_now - DAY_ATRA_START)} der Induktion)"

    title_grid.set_text(f"Tag {day_now:.1f} | {phase}")

    for bar, k in zip(bars, [HSC, GMP, PROMYELO, MYELO, GRANULO]):
        bar.set_height(counts[k])

    early = [h + g for h, g in zip(history[HSC], history[GMP])]
    mature = [m + gr for m, gr in zip(history[MYELO], history[GRANULO])]

    l_prog.set_data(history["day"], early)
    l_prom.set_data(history["day"], history[PROMYELO])
    l_mat.set_data(history["day"], mature)

    return img, *bars, l_prog, l_prom, l_mat, title_grid


total_frames = MAX_TICKS // 2
ani = FuncAnimation(fig, update, frames=total_frames, interval=20, blit=False, repeat=False)
plt.show()