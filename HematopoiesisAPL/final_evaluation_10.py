import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. KONFIGURATION FÜR DIE 10-FACHE STATISTISCHE AUSWERTUNG
# ==============================================================================
N_RUNS = 10                  # 10 unabhängige Simulationsläufe (Seeds 0 bis 9)
GRID_H, GRID_W = 40, 50      # 2.000 Plätze im Knochenmark
TOTAL_TICKS = 750
TICK_MUTATION = 180          # Auftreten von PML-RARα
TICK_ATRA = 420              # Therapiestart mit ATRA
INITIAL_HSCS = 15            # 15 Ursprungs-Stammzell-Klone

# Reifungsstufen
HSC = 1
GMP = 2
PROMYELO = 3
MYELO = 4
GRANULO = 5
APL_CLONE_ID = -1


# ==============================================================================
# 2. MODELL-DEFINITION (5 Stufen + Klonales Tracking)
# ==============================================================================
class MyeloidCell:
    def __init__(self, stage, x, y, clone_id, is_leukemic=False):
        self.stage = stage
        self.x = x
        self.y = y
        self.clone_id = clone_id
        self.is_leukemic = is_leukemic
        self.age = 0
        self.divisions = 0
        self.atra_sensitivity = np.random.uniform(0.75, 1.25)

    def step(self, grid, atra_active, atra_dose):
        self.age += 1

        if self.stage == HSC:
            if np.random.rand() < 0.002:
                grid[self.x, self.y] = None
                return
            if self.age >= 10 and np.random.rand() < 0.12:
                free = get_free_neighbors(grid, self.x, self.y, bias_right=True)
                if free:
                    nx, ny = free[np.random.randint(len(free))]
                    child_stage = GMP if np.random.rand() < 0.90 else HSC
                    grid[nx, ny] = MyeloidCell(child_stage, nx, ny, clone_id=self.clone_id)
                    self.age = 0

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
                    child = MyeloidCell(GMP, nx, ny, clone_id=self.clone_id)
                    child.divisions = self.divisions + 1
                    grid[nx, ny] = child
                    self.divisions += 1
                    self.age = 0

        elif self.stage == PROMYELO:
            if self.is_leukemic:
                if not atra_active:
                    if np.random.rand() < 0.006:
                        grid[self.x, self.y] = None
                        return
                    if self.age >= 6 and np.random.rand() < 0.30:
                        free = get_free_neighbors(grid, self.x, self.y, bias_right=False)
                        if free:
                            nx, ny = free[np.random.randint(len(free))]
                            grid[nx, ny] = MyeloidCell(PROMYELO, nx, ny, clone_id=APL_CLONE_ID, is_leukemic=True)
                            self.age = 0
                else:
                    if self.age >= 4 and np.random.rand() < (atra_dose * self.atra_sensitivity):
                        self.stage = MYELO
                        self.is_leukemic = False
                        self.age = 0
                        return
                    if self.age >= 8 and np.random.rand() < 0.04:
                        free = get_free_neighbors(grid, self.x, self.y, bias_right=True)
                        if free:
                            nx, ny = free[np.random.randint(len(free))]
                            grid[nx, ny] = MyeloidCell(PROMYELO, nx, ny, clone_id=APL_CLONE_ID, is_leukemic=True)
                            self.age = 0
            else:
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
                        child = MyeloidCell(PROMYELO, nx, ny, clone_id=self.clone_id)
                        child.divisions = self.divisions + 1
                        grid[nx, ny] = child
                        self.divisions += 1
                        self.age = 0

        elif self.stage == MYELO:
            if np.random.rand() < 0.01:
                grid[self.x, self.y] = None
                return
            if self.age >= 8 and np.random.rand() < 0.28:
                self.stage = GRANULO
                self.age = 0

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
            if 0 <= nx < GRID_H and 0 <= ny < GRID_W and grid[nx, ny] is None:
                free_spots.append((nx, ny))
                if bias_right and dy == 1:
                    free_spots.append((nx, ny))
    return free_spots


def run_single_simulation(seed, atra_dose=0.28):
    """Führt einen kompletten Simulationslauf durch und gibt alle Zeitreihen zurück."""
    np.random.seed(seed)
    grid = np.empty((GRID_H, GRID_W), dtype=object)

    for clone_idx in range(1, INITIAL_HSCS + 1):
        while True:
            rx, ry = np.random.randint(0, GRID_H), np.random.randint(0, 8)
            if grid[rx, ry] is None:
                grid[rx, ry] = MyeloidCell(HSC, rx, ry, clone_id=clone_idx)
                break

    ts = {
        "early": np.zeros(TOTAL_TICKS),        # HSC + GMP
        "promyelo": np.zeros(TOTAL_TICKS),     # Promyelozyten (inkl. APL-Blasten)
        "granulo": np.zeros(TOTAL_TICKS),      # Reife Granulozyten (Stufe 5)
        "active_clones": np.zeros(TOTAL_TICKS),# Anzahl aktiver gesunder Ursprungs-Klone
        "stages_health": np.zeros(5),          # Schnappschuss Tick 179 (Gesund)
        "stages_apl": np.zeros(5),             # Schnappschuss Tick 419 (APL)
        "stages_remission": np.zeros(5)        # Schnappschuss Tick 749 (Remission)
    }

    for tick in range(TOTAL_TICKS):
        atra_active = (tick >= TICK_ATRA) and (atra_dose > 0)

        if tick == TICK_MUTATION:
            candidates = [(x, y) for x in range(GRID_H) for y in range(GRID_W)
                          if grid[x, y] is not None and grid[x, y].stage in [GMP, PROMYELO]]
            if len(candidates) >= 4:
                for idx in np.random.choice(len(candidates), 4, replace=False):
                    cx, cy = candidates[idx]
                    grid[cx, cy] = MyeloidCell(PROMYELO, cx, cy, clone_id=APL_CLONE_ID, is_leukemic=True)

        agents = [grid[x, y] for x in range(GRID_H) for y in range(GRID_W) if grid[x, y] is not None]
        np.random.shuffle(agents)
        for agent in agents:
            if grid[agent.x, agent.y] is agent:
                agent.step(grid, atra_active, atra_dose)

        stage_counts = [0, 0, 0, 0, 0]
        healthy_clones_present = set()

        for x in range(GRID_H):
            for y in range(GRID_W):
                cell = grid[x, y]
                if cell is not None:
                    stage_counts[cell.stage - 1] += 1
                    if cell.clone_id != APL_CLONE_ID:
                        healthy_clones_present.add(cell.clone_id)

        ts["early"][tick] = stage_counts[0] + stage_counts[1]
        ts["promyelo"][tick] = stage_counts[2]
        ts["granulo"][tick] = stage_counts[4]
        ts["active_clones"][tick] = len(healthy_clones_present)

        if tick == TICK_MUTATION - 1:
            ts["stages_health"] = np.array(stage_counts)
        elif tick == TICK_ATRA - 1:
            ts["stages_apl"] = np.array(stage_counts)
        elif tick == TOTAL_TICKS - 1:
            ts["stages_remission"] = np.array(stage_counts)

    return ts


# ==============================================================================
# 3. HAUPTPROGRAMM: 10 DURCHLÄUFE + DOSIS-VERGLEICH + AUSWERTUNG
# ==============================================================================
if __name__ == "__main__":
    print(f"Starte {N_RUNS} Durchläufe für Standard-Therapie (ATRA = 0.28)...")
    runs_std = [run_single_simulation(seed=s, atra_dose=0.28) for s in range(N_RUNS)]

    print(f"Starte {N_RUNS} Durchläufe für subtherapeutische Dosis (ATRA = 0.05)...")
    runs_low = [run_single_simulation(seed=s, atra_dose=0.05) for s in range(N_RUNS)]

    # Hilfsfunktion für Mittelwert und Standardabweichung
    def get_stats(runs, key):
        mat = np.array([r[key] for r in runs])
        return np.mean(mat, axis=0), np.std(mat, axis=0)

    ticks = np.arange(TOTAL_TICKS)

    # Zeitreihen-Statistiken berechnen
    early_m, early_s = get_stats(runs_std, "early")
    prom_m, prom_s = get_stats(runs_std, "promyelo")
    gran_m, gran_s = get_stats(runs_std, "granulo")
    clones_m, clones_s = get_stats(runs_std, "active_clones")

    prom_low_m, prom_low_s = get_stats(runs_low, "promyelo")

    # Stufen-Verteilung für die 3 Phasen (Mittelwert ± SD)
    st_h_m, st_h_s = get_stats(runs_std, "stages_health")
    st_a_m, st_a_s = get_stats(runs_std, "stages_apl")
    st_r_m, st_r_s = get_stats(runs_std, "stages_remission")

    # ==========================================================================
    # 4. EXAKTE ZAHLENWERTE FÜR DEIN PROTOKOLL IM TERMINAL AUSGEBEN
    # ==========================================================================
    print("\n" + "=" * 65)
    print(f"STATISTISCHE AUSWERTUNG ÜBER N = {N_RUNS} SIMULATIONS-LÄUFE (MEAN ± SD)")
    print("=" * 65)
    print(f"1. Gesunde Homöostase (Tick {TICK_MUTATION - 1}):")
    print(f"   - Reife Granulozyten (Stufe 5): {st_h_m[4]:.1f} ± {st_h_s[4]:.1f} Zellen")
    print(f"   - Aktive gesunde Klone:         {clones_m[TICK_MUTATION - 1]:.1f} ± {clones_s[TICK_MUTATION - 1]:.1f} / {INITIAL_HSCS}")

    print(f"\n2. Unbehandelte APL-Phase (Tick {TICK_ATRA - 1}):")
    print(f"   - Promyelozyten (Block Stufe 3): {st_a_m[2]:.1f} ± {st_a_s[2]:.1f} Zellen")
    print(f"   - Reife Granulozyten (Stufe 5):  {st_a_m[4]:.1f} ± {st_a_s[4]:.1f} Zellen")
    print(f"   - Aktive gesunde Klone:          {clones_m[TICK_ATRA - 1]:.1f} ± {clones_s[TICK_ATRA - 1]:.1f} / {INITIAL_HSCS}")

    peak_gran_idx = TICK_ATRA + np.argmax(gran_m[TICK_ATRA:TICK_ATRA + 80])
    print(f"\n3. ATRA-Therapie & Remission (Tick {TOTAL_TICKS - 1}):")
    print(f"   - Differenzierungssyndrom-Peak:  {gran_m[peak_gran_idx]:.1f} ± {gran_s[peak_gran_idx]:.1f} Granulozyten (bei Tick {peak_gran_idx})")
    print(f"   - Promyelozyten am Ende:         {st_r_m[2]:.1f} ± {st_r_s[2]:.1f} Zellen")
    print(f"   - Reife Granulozyten am Ende:    {st_r_m[4]:.1f} ± {st_r_s[4]:.1f} Zellen")
    print(f"   - Überlebende gesunde Klone:     {clones_m[-1]:.1f} ± {clones_s[-1]:.1f} / {INITIAL_HSCS} (Klonales Bottleneck!)")
    print("=" * 65 + "\n")

    # ==========================================================================
    # 5. PUBLIKATIONSREIFES 4-TEILIGES AUSWERTUNGSDIAGRAMM
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    fig.suptitle(f"Statistische Gesamtauswertung (N = {N_RUNS} Simulationsläufe, Mittelwert ± SD)", fontsize=14, fontweight="bold")

    # --- PANEL A (Oben Links): Populationsdynamik über die 3 Phasen ---
    ax = axes[0, 0]
    ax.plot(ticks, early_m, label="Frühe Vorläufer (HSC + GMP)", color="#33a02c", lw=2)
    ax.fill_between(ticks, early_m - early_s, early_m + early_s, color="#33a02c", alpha=0.18)

    ax.plot(ticks, prom_m, label="Promyelozyten (Stufe 3 / APL)", color="#e31a1c", lw=2.3)
    ax.fill_between(ticks, prom_m - prom_s, prom_m + prom_s, color="#e31a1c", alpha=0.18)

    ax.plot(ticks, gran_m, label="Reife Granulozyten (Stufe 5)", color="#00b4d8", lw=2)
    ax.fill_between(ticks, gran_m - gran_s, gran_m + gran_s, color="#00b4d8", alpha=0.18)

    ax.axvline(TICK_MUTATION, color="black", ls=":", lw=1.5)
    ax.axvline(TICK_ATRA, color="darkorange", ls="--", lw=1.8)
    ax.set_title("A) Zelldynamik: Homöostase, APL-Block & ATRA-Remission", fontweight="bold")
    ax.set_xlabel("Zeit (Ticks / Stunden)")
    ax.set_ylabel("Zellzahl (Mean ± SD)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)

    # --- PANEL B (Oben Rechts): Vergleich der 5 Reifungsstufen in den 3 Phasen ---
    ax = axes[0, 1]
    x_pos = np.arange(5)
    width = 0.26
    stage_labels = ["1. HSC", "2. GMP", "3. Promyelo", "4. Myelo", "5. Granulo"]

    ax.bar(x_pos - width, st_h_m, width, yerr=st_h_s, capsize=3, label=f"Gesund (Tick {TICK_MUTATION - 1})", color="#2b8cbe")
    ax.bar(x_pos, st_a_m, width, yerr=st_a_s, capsize=3, label=f"APL Blockiert (Tick {TICK_ATRA - 1})", color="#e31a1c")
    ax.bar(x_pos + width, st_r_m, width, yerr=st_r_s, capsize=3, label=f"ATRA Remission (Tick {TOTAL_TICKS - 1})", color="#33a02c")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stage_labels)
    ax.set_title("B) Reifungs-Spektrum der 5 Differenzierungsstufen", fontweight="bold")
    ax.set_ylabel("Zellzahl (Mean ± SD)")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    # --- PANEL C (Unten Links): Klonale Diversität & Flaschenhals (Bottleneck) ---
    ax = axes[1, 0]
    ax.plot(ticks, clones_m, color="#6a3d9a", lw=2.3, label="Aktive gesunde Stammzell-Klone")
    ax.fill_between(ticks, clones_m - clones_s, clones_m + clones_s, color="#6a3d9a", alpha=0.2)
    ax.axvline(TICK_MUTATION, color="black", ls=":", lw=1.5, label="APL-Mutation")
    ax.axvline(TICK_ATRA, color="darkorange", ls="--", lw=1.8, label="ATRA-Start")

    ax.set_title("C) Klonale Diversität & Therapie-Bottleneck (von 15 Start-Klonen)", fontweight="bold")
    ax.set_xlabel("Zeit (Ticks / Stunden)")
    ax.set_ylabel("Anzahl aktiver gesunder Klone")
    ax.set_ylim(0, INITIAL_HSCS + 1)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(alpha=0.25)

    # --- PANEL D (Unten Rechts): Dosis-Wirkungs-Vergleich (Schwellenwert) ---
    ax = axes[1, 1]
    ax.plot(ticks, prom_low_m, label="Subtherapeutische ATRA-Dosis (0.05)", color="#ff7f00", lw=2.2)
    ax.fill_between(ticks, prom_low_m - prom_low_s, prom_low_m + prom_low_s, color="#ff7f00", alpha=0.2)

    ax.plot(ticks, prom_m, label="Therapeutische ATRA-Dosis (0.28)", color="#e31a1c", lw=2.2)
    ax.fill_between(ticks, prom_m - prom_s, prom_m + prom_s, color="#e31a1c", alpha=0.2)

    ax.axvline(TICK_ATRA, color="darkorange", ls="--", lw=1.8, label="ATRA-Start")
    ax.set_title("D) Sensitivitätsanalyse: Promyelozyten unter zwei ATRA-Dosen", fontweight="bold")
    ax.set_xlabel("Zeit (Ticks / Stunden)")
    ax.set_ylabel("Anzahl Promyelozyten (Mean ± SD)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("auswertung_10_laeufe.png", dpi=300)  # Speichert Grafik direkt hochauflösend!
    plt.show()