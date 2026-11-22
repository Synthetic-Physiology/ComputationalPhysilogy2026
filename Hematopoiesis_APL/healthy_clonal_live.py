import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

# ==============================================================================
# 1. KONFIGURATION (Rein physiologische Myelopoese – keine APL, keine Therapie)
# ==============================================================================
GRID_H, GRID_W = 40, 50      # 2.000 Plätze im Knochenmark
TOTAL_TICKS = 600            # Simulationsdauer in Stunden (Ticks)
INITIAL_HSCS = 15            # 15 gesunde Stammzellen mit eigener Klon-ID (1 bis 15)

# Reifungsstufen
HSC = 1
GMP = 2
PROMYELO = 3
MYELO = 4
GRANULO = 5


# ==============================================================================
# 2. GESUNDER ZELL-AGENT MIT VERERBBARER KLON-ID
# ==============================================================================
class HealthyClonalCell:
    def __init__(self, stage, x, y, clone_id):
        self.stage = stage
        self.x = x
        self.y = y
        self.clone_id = clone_id      # Wird bei Teilung an Tochterzellen vererbt
        self.age = 0
        self.divisions = 0

    def step(self, grid):
        self.age += 1

        # 1. STAMMZELLE (HSC)
        if self.stage == HSC:
            if np.random.rand() < 0.002:
                grid[self.x, self.y] = None
                return
            if self.age >= 10 and np.random.rand() < 0.13:
                free = get_free_neighbors(grid, self.x, self.y, bias_right=True)
                if free:
                    nx, ny = free[np.random.randint(len(free))]
                    # 88% asymmetrisch (Vorläufer), 12% symmetrisch (neue Stammzelle)
                    child_stage = GMP if np.random.rand() < 0.88 else HSC
                    grid[nx, ny] = HealthyClonalCell(child_stage, nx, ny, clone_id=self.clone_id)
                    self.age = 0

        # 2. PROGENITOR (GMP)
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
                    child = HealthyClonalCell(GMP, nx, ny, clone_id=self.clone_id)
                    child.divisions = self.divisions + 1
                    grid[nx, ny] = child
                    self.divisions += 1
                    self.age = 0

        # 3. PROMYELOZYT
        elif self.stage == PROMYELO:
            if np.random.rand() < 0.01:
                grid[self.x, self.y] = None
                return
            if self.age >= 8 and np.random.rand() < 0.28:
                self.stage = MYELO
                self.age = 0
                return
            if self.divisions < 5 and self.age >= 7 and np.random.rand() < 0.16:
                free = get_free_neighbors(grid, self.x, self.y, bias_right=True)
                if free:
                    nx, ny = free[np.random.randint(len(free))]
                    child = HealthyClonalCell(PROMYELO, nx, ny, clone_id=self.clone_id)
                    child.divisions = self.divisions + 1
                    grid[nx, ny] = child
                    self.divisions += 1
                    self.age = 0

        # 4. MYELOZYT / METAMYELOZYT
        elif self.stage == MYELO:
            if np.random.rand() < 0.01:
                grid[self.x, self.y] = None
                return
            if self.age >= 8 and np.random.rand() < 0.28:
                self.stage = GRANULO
                self.age = 0

        # 5. REIFER GRANULOZYT
        elif self.stage == GRANULO:
            if self.age >= 16 and np.random.rand() < 0.25:
                grid[self.x, self.y] = None


def get_free_neighbors(grid, x, y, bias_right=False):
    free_spots = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_H and 0 <= ny < GRID_W:
                if grid[nx, ny] is None:
                    free_spots.append((nx, ny))
                    if bias_right and dy == 1:
                        free_spots.append((nx, ny))
    return free_spots


# ==============================================================================
# 3. INITIALISIERUNG & LIVE-ANIMATION
# ==============================================================================
np.random.seed(42)
grid = np.empty((GRID_H, GRID_W), dtype=object)

# 15 initiale Stammzellen mit eindeutiger Klon-ID (1 bis 15) links platzieren
for clone_idx in range(1, INITIAL_HSCS + 1):
    while True:
        rx = np.random.randint(0, GRID_H)
        ry = np.random.randint(0, 8)
        if grid[rx, ry] is None:
            grid[rx, ry] = HealthyClonalCell(HSC, rx, ry, clone_id=clone_idx)
            break

clone_ids = list(range(1, INITIAL_HSCS + 1))
clone_history = {cid: [] for cid in clone_ids}
recorded_ticks = []

# Farbpalette für die 15 gesunden Stammzell-Linien
cmap_base = plt.get_cmap("tab20", INITIAL_HSCS)
clone_colors = [cmap_base(i) for i in range(INITIAL_HSCS)]

grid_colors = ["#f5f5f5"] + clone_colors
grid_cmap = ListedColormap(grid_colors)

# Layout: Links Klon-Territorien, Rechts Live-Muller-Plot (Prozentual)
fig, (ax_grid, ax_muller) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.45]})
fig.suptitle("Physiologische Hämatopoese: Polyklonale Homöostase & Neutraler Klonaler Drift", fontsize=13, fontweight="bold")

img = ax_grid.imshow(np.zeros((GRID_H, GRID_W)), cmap=grid_cmap, vmin=0, vmax=INITIAL_HSCS)
ax_grid.set_xticks([])
ax_grid.set_yticks([])
ax_grid.set_xlabel("Jede Farbe = Nachkommen einer eigenen Stammzelle (HSC 1–15)", fontsize=8.5, fontweight="bold")
title_grid = ax_grid.set_title("Klon-Territorien (Tick 0)", fontweight="bold")

ax_muller.set_xlim(0, TOTAL_TICKS)
ax_muller.set_ylim(0, 100)
ax_muller.set_xlabel("Simulationszeit (Ticks / Stunden)")
ax_muller.set_ylabel("Klonaler Anteil an der Gesamtpopulation (%)")
ax_muller.set_title("Live Muller-Plot (Polyklonale Verteilung)", fontweight="bold")


def update(frame):
    agents = [grid[x, y] for x in range(GRID_H) for y in range(GRID_W) if grid[x, y] is not None]
    np.random.shuffle(agents)
    for agent in agents:
        if grid[agent.x, agent.y] is agent:
            agent.step(grid)

    visual_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    counts = {cid: 0 for cid in clone_ids}

    for x in range(GRID_H):
        for y in range(GRID_W):
            if grid[x, y] is not None:
                cid = grid[x, y].clone_id
                counts[cid] += 1
                visual_grid[x, y] = cid

    recorded_ticks.append(frame)
    for cid in clone_ids:
        clone_history[cid].append(counts[cid])

    # Linkes Gitter aktualisieren
    img.set_data(visual_grid)
    active_clones = sum(1 for cid in clone_ids if counts[cid] > 0)
    total_cells = sum(counts.values())
    title_grid.set_text(f"Klon-Territorien (Tick {frame} | Aktive Klone: {active_clones}/{INITIAL_HSCS} | Zellen: {total_cells})")

    # Rechten Muller-Plot alle 3 Ticks aktualisieren
    if frame % 3 == 0 or frame == TOTAL_TICKS - 1:
        ax_muller.clear()
        y_stacks = [np.array(clone_history[cid]) for cid in clone_ids]
        totals = np.sum(y_stacks, axis=0)
        totals[totals == 0] = 1
        y_rel = [(y / totals) * 100 for y in y_stacks]

        ax_muller.stackplot(recorded_ticks, y_rel, colors=clone_colors, alpha=0.9)
        ax_muller.set_xlim(0, TOTAL_TICKS)
        ax_muller.set_ylim(0, 100)
        ax_muller.set_xlabel("Simulationszeit (Ticks / Stunden)")
        ax_muller.set_ylabel("Klonaler Anteil (%)")
        ax_muller.set_title("Live Muller-Plot (Polyklonale Verteilung)", fontweight="bold")

    return img, title_grid


ani = FuncAnimation(fig, update, frames=range(TOTAL_TICKS), interval=20, blit=False, repeat=False)
plt.tight_layout()
plt.show()