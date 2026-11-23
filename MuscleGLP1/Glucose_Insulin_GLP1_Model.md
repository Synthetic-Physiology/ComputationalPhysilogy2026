# Glucosemetabolismus

Modellcode: GLP-1, Muskel und Fett · Python 3 (numpy, scipy, matplotlib)

[READMEÜberblick und Reihenfolge](#readme)

Modell

[model.pyModellkern: Zustände, Flüsse, oGTT](#model-py)[calibrate.pyKalibrierung der gesunden Referenz](#calibrate-py)[groups.pyGruppen aus Literaturfaktoren](#groups-py)[intervention.pyFettverlust, Magermasse, GLP-1, Training](#intervention-py)[scenarios.pySzenarien und Shapley-Zerlegung](#scenarios-py)[mc.pyUnsicherheitsanalyse (400 Varianten)](#mc-py)[mc_summary.pyZusammenfassung der Unsicherheit](#mc_summary-py)[relevance.pyParameter-Relevanz ±20 %](#relevance-py)

Abbildungen

[figures.pyHauptabbildungen](#figures-py)[fig_pot_ins.pyPotenzial Insulin](#fig_pot_ins-py)[fig_valid2.pyValidierung](#fig_valid2-py)[fig_glp_groups.pyGLP-1 in vier Gruppen](#fig_glp_groups-py)[fig_base_groups.pyAusgangslage vier Gruppen](#fig_base_groups-py)

## README

Überblick und Reihenfolge

Wahlfach Computational Physiology. Python 3 mit numpy, scipy, matplotlib.

### Dateien

| Datei | Inhalt |
| --- | --- |
| `model.py` | Modellkern: 5 Zustände (Magen, Darm, Glukose, Insulin, wirksames Insulin), Flüsse, Nüchternzustand, oGTT-Simulation |
| `calibrate.py` | Kalibrierung der gesunden Referenz (Maximum a posteriori), schreibt `calibration.json` |
| `groups.py` | Gruppen gesund, adipös NGT, adipös IGT, T2D aus Literaturfaktoren |
| `intervention.py` | Interventionen: Fettverlust (Magkos 2016), Magermasse, GLP-1-Direktwirkungen, Krafttraining (+1,0 kg, ×1,15) |
| `scenarios.py` | Szenarien und Shapley-Zerlegung, schreibt `scenarios.json` |
| `mc.py`, `mc_summary.py` | Unsicherheitsanalyse mit 400 Modellvarianten, schreibt `mc.json`, `mc_summary.json` |
| `relevance.py` | Sensitivität jedes Parameters (±20 %) |
| `figures.py`, `fig_*.py` | Abbildungen für die Präsentation (Ordner `fig/`) |
| `calibration.json` | kalibrierte Parameter mit Kovarianz |
| `gut_cache.json` | vorberechnete Magenparameter (wird sonst beim ersten Start erzeugt) |
| `simulator/index.html` | interaktiver Simulator (JavaScript-Port des Modells, im Browser öffnen) |

### Reihenfolge

```
python calibrate.py        # optional, calibration.json liegt bei
python scenarios.py
python mc.py 400           # ca. 8 min
python mc_summary.py
python figures.py
```

Beispiel einer einzelnen Rechnung:

```
import model as m, intervention as iv
q = iv.make("adipös IGT", **iv.STEP1, drug=iv.GLP1)   # Semaglutid wie STEP 1
t, y = m.ogtt(q, 180)                                  # y[2] Glukose (mmol/L), y[3] Insulin (pmol/L)
```

## model.py

Modellkern: Zustände, Flüsse, oGTT · 170 Zeilen

```python
"""
Glukose-Insulin-Modell zur Forschungsfrage Muskel- vs. Fettverlust unter GLP-1 (finale Version).

Zustaende (5): S Magen [g], Q Darm [g], G Plasmaglukose [mmol/L], I Plasmainsulin [pmol/L],
               X wirksames Insulin [pmol/L] (verzoegert, wirkt auf Aufnahme und Leber).
Fluesse in mmol/min. Insulinsekretion pro m2 Koerperoberflaeche (kuerzt sich mit der Clearance pro m2).

Herkunft: [L] Literatur, [B] Bilanz/abgeleitet, [A] begruendete Annahme, [K] kalibriert innerhalb Literaturstreuung.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve

MW = 180.16

# ------------------------------------------------------------------ Grundparameter (gesunde Referenz)
LIT = dict(
    BW=70.0, FFM=52.0, FM=18.0,      # [A] Referenzperson RISC (44 J., BMI ~25)
    Gb=4.95, Ib=38.0,                # [L] Median-Referenz nuechtern (RISC 5,1 / 35)
    V_dl=1.45,                       # [L] Glukose-Verteilungsvolumen dl/kg (Dalla Man 2004)
    EGP_b=2.0,                       # [L] mg/kg/min basale Leberproduktion (Ferrannini 1985: 1,9-2,3)
    f_ii=0.75,                       # [L] insulinunabhaengiger Anteil der basalen Aufnahme (Ferrannini 1985 JCI)
    U_brain_gd=120.0,                # [L] g/Tag Hirnaufnahme (Owen 1967), gesaettigt -> konstant
    SG=0.025,                        # [L/K] Glukoseeffektivitaet min^-1 (orales Minimalmodell, typ. 0,015-0,035)
    SI_D=9.93e-4,                    # [L/K] dl/kg/min pro uU/ml: Insulinwirkung auf Aufnahme bei basalem Insulin (Visentin 2015)
    SI_P=5.48e-4,                    # [L] Leber (Visentin 2015) - nur zur Pruefung der Steigung
    X50_D=330.0,                     # [L/K] pmol/L halbmax. insulinabh. Aufnahme (Rizza 1981: 55 uU/ml)
    X50_L=71.0,                      # [L/K] pmol/L halbmax. EGP-Hemmung, peripher-aequivalent (Rizza 29 uU/ml / Portalfaktor 2,45 Nurjhan)
    sP=0.5,                          # [A/K] Anteil der Glukoseeffektivitaet ueber Hemmung der Leber
    p2=0.02,                         # [L/K] min^-1 Wirkverzoegerung Insulin (OMM typ. 0,01-0,04)
    musc_SI=0.80,                    # [L] Muskelanteil insulinabh. Aufnahme (DeFronzo; Honka 2018)
    musc_ii=0.30,                    # [A] Muskelanteil insulinunabh. Aufnahme ausser Hirn
    CL=1.6,                          # [L/K] L/min/m2 endogene Insulinclearance im oGTT (RISC)
    t_half_I=4.3,                    # [L] min Plasma-Halbwertszeit Insulin
    beta=119.0,                      # [L/K] pmol/min/m2/mM Glukosesensitivitaet Betazelle (RISC NGT)
    RS=800.0,                        # [L/K] pmol/m2/mM Ratensensitivitaet (Mari-Modell, NGT 680-910)
    PFR=2.0,                         # [L/K] Potenzierung 2 h vs basal (Ferrannini 2003)
    D=75.0, k_abs=0.057,             # [L] g Glukose; Darmresorption min^-1 (Dalla Man 2007)
    GE_max=0.5,                      # [L/K] g/min max. Magenentleerung (2 kcal/min; Bereich 1-4 kcal/min)
    G_thr=10.0, GFR=0.12,            # [L] Nierenschwelle mmol/L, GFR L/min
    w_glucagon=0.6,                  # [A] Anteil der basalen EGP, der von Glukagon abhaengt (Cherrington)
)

# Magen: K und systemischer Anteil f werden fuer jedes GE_max aus zwei Tracer-Ankerwerten berechnet [B]:
# Ferrannini 1985: 72 % der Last bis 210 min systemisch erschienen; Ra(210) = 2,47 mg/kg/min.
def _ra_at(GE, K, f, kab, D, BW, T=210):
    def r(t, y):
        S, Q, C = y; ge = GE * S / (K + S); return [-ge, ge - kab * Q, f * kab * Q]
    s = solve_ivp(r, (0, T), [D, 0, 0], rtol=1e-9, atol=1e-12)
    return f * kab * s.y[1, -1] * 1000 / BW, s.y[2, -1] / D
import os, json as _json
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gut_cache.json")
_GE_GRID = np.linspace(0.40, 0.72, 17); _KA_GRID = np.array([0.03, 0.04, 0.057, 0.08, 0.11, 0.15, 0.2])
if os.path.exists(_CACHE):
    _KF = np.array(_json.load(open(_CACHE)))
else:
    _KF = np.zeros((len(_KA_GRID), len(_GE_GRID), 2))
    for i, _ka in enumerate(_KA_GRID):
        z = [20, 0.85]
        for j, _ge in enumerate(_GE_GRID):
            z = fsolve(lambda z: [_ra_at(_ge, abs(z[0]), z[1], _ka, 75, 70)[0] - 2.47,
                                  _ra_at(_ge, abs(z[0]), z[1], _ka, 75, 70)[1] - 0.72], z)
            _KF[i, j] = (abs(z[0]), z[1])
    _json.dump(_KF.tolist(), open(_CACHE, "w"))
def gut_params(GE, ka=0.057):
    from scipy.interpolate import RegularGridInterpolator
    out = []
    for c in (0, 1):
        f = RegularGridInterpolator((_KA_GRID, _GE_GRID), _KF[:, :, c])
        out.append(float(f([[np.clip(ka, _KA_GRID[0], _KA_GRID[-1]), np.clip(GE, _GE_GRID[0], _GE_GRID[-1])]])[0]))
    return out[0], out[1]

# ------------------------------------------------------------------ Aufbau einer Person
DEFAULT_PERSON = dict(
    BW=70.0, FFM=52.0, FM=18.0,      # Koerperzusammensetzung
    Gb=None, Ib=None,                # Nuechternanker (None = Referenz); bei Interventionen endogen geloest
    q_m=1.0, q_r=1.0, q_L=1.0,       # Insulinsensitivitaet je Masse: Muskel, uebrige Gewebe, Leber
    cl_f=1.0,                        # Faktor Insulinclearance
    M_rel=None,                      # Muskelmasse relativ zur Referenz (None: wie FFM)
    beta_f=1.0, RS_f=1.0, PFR=None,  # Betazelle relativ zur Referenz
    drug_isr=1.0, drug_glucagon=1.0, drug_ge=1.0,   # GLP-1-Direkteffekte (1 = keine)
)

def build(par, person=None):
    """par: Grundparameter (LIT ggf. kalibriert). person: dict nach DEFAULT_PERSON."""
    P = dict(DEFAULT_PERSON); P.update(person or {})
    p = dict(par)
    V0 = p["V_dl"] * p["BW"] / 10                           # L, Referenz
    ffm_rel = P["FFM"] / p["FFM"]
    q = dict(p); q["person"] = P
    q["V"] = V0 * ffm_rel                                    # [A] Volumen ~ extrazellulaeres Wasser ~ FFM
    si = lambda s: s / p["V_dl"] / 6.0 * V0                  # dl/kg/min/(uU/ml) -> L/min pro pmol/L
    q["U_brain"] = p["U_brain_gd"] / MW / 1440 * 1000        # mmol/min
    EGPb = p["EGP_b"] * p["BW"] / MW                         # mmol/min (Referenz)
    k_ii0 = (p["f_ii"] * EGPb - q["U_brain"]) / p["Gb"]      # [B] L/min
    m_rel_ = P.get("M_rel") or ffm_rel
    q["k_ii"] = k_ii0 * ((1 - p["musc_ii"]) * ffm_rel + p["musc_ii"] * m_rel_)
    q["k_ii_m"] = k_ii0 * p["musc_ii"] * m_rel_
    # insulinabh. Aufnahme: cl_D(X) = cl_max * X/(X50_D + X); Steigung bei Ib = SI_D (Visentin) [B]
    clmax = si(p["SI_D"]) * (p["X50_D"] + p["Ib"]) ** 2 / p["X50_D"]
    m_rel = P.get("M_rel") or ffm_rel                                           # Muskelmasse relativ (Standard ~ FFM) [A]
    q["clmax_m"] = clmax * p["musc_SI"] * m_rel * P["q_m"]
    q["clmax_r"] = clmax * (1 - p["musc_SI"]) * P["q_r"]
    q["qL"] = P["q_L"]
    # Glukoseeffekt auf die Leber [A/K]: Anteil sP der Glukoseeffektivitaet SG*V
    q["k_GP"] = p["sP"] * p["SG"] * V0
    q["CL"] = p["CL"] * P["cl_f"]
    q["beta"] = p["beta"] * P["beta_f"]; q["RS"] = p["RS"] * P["RS_f"]
    q["PFR"] = P["PFR"] if P["PFR"] is not None else p["PFR"]
    q["n"] = np.log(2) / p["t_half_I"]
    q["K_ge"], q["f"] = gut_params(p["GE_max"], p["k_abs"])
    q["GE"] = p["GE_max"] * P["drug_ge"]
    # Leber: E0 aus Nuechternbilanz am Anker (Gruppe) [B]; Glukagon-Anteil w
    Ga = P["Gb"] if P["Gb"] is not None else p["Gb"]; Ia = P["Ib"] if P["Ib"] is not None else p["Ib"]
    q["Ga"], q["Ia"] = Ga, Ia
    for k in ("X50_D", "X50_L", "GFR", "G_thr"): q[k] = p[k]
    U = uptake(Ga, Ia, q)
    q["E0"] = (U + q["k_GP"] * Ga) / hep(Ia, q)                              # EGP(Ga,Ia) = U  (ohne Medikament)
    q["glucagon"] = 1 - p["w_glucagon"] * (1 - P["drug_glucagon"])
    # Betazelle: ISR(G) = ISR5 + beta*(G-5); ISR5 aus Anker: CL*Ia = ISR(Ga) [B]
    q["ISR5"] = q["CL"] * Ia - q["beta"] * (Ga - 5.0)
    q["drug_isr"] = P["drug_isr"]
    return q

def cl_D(X, q, part="all"):
    h = max(X, 0.0) / (q["X50_D"] + max(X, 0.0))
    return h * {"all": q["clmax_m"] + q["clmax_r"], "m": q["clmax_m"]}[part]

def uptake(G, X, q):
    return q["U_brain"] + q["k_ii"] * G + cl_D(X, q) * G + q["GFR"] * max(G - q["G_thr"], 0.0)

def hep(X, q):
    return 1.0 / (1.0 + q["qL"] * max(X, 0.0) / q["X50_L"])

def egp(G, X, q):
    return max(q["E0"] * q["glucagon"] * hep(X, q) - q["k_GP"] * G, 0.0)

def isr_basal(G, q):
    return q["drug_isr"] * max(q["ISR5"] + q["beta"] * (G - 5.0), 0.0)

def fasting(q):
    """Nuechterngleichgewicht (G, I): EGP = Aufnahme, I = ISR/CL."""
    f = lambda G: egp(G, isr_basal(G, q) / q["CL"], q) - uptake(G, isr_basal(G, q) / q["CL"], q)
    G = brentq(f, 2.5, 25.0); return G, isr_basal(G, q) / q["CL"]

def rhs(t, y, q):
    S, Qg, G, I, X = y
    ge = q["GE"] * S / (q["K_ge"] + S)
    Ra = q["f"] * q["k_abs"] * Qg * 1000 / MW
    dG = (egp(G, X, q) + Ra - uptake(G, X, q)) / q["V"]
    P = 1 + (q["PFR"] - 1) * min(t / 120.0, 1.0)            # Potenzierung (Mari), oGTT-Zeitverlauf
    ISR = P * isr_basal(G, q) + q["drug_isr"] * q["RS"] * max(dG, 0.0)
    return [-ge, ge - q["k_abs"] * Qg, dG, q["n"] * (ISR / q["CL"] - I), q["p2"] * (I - X)]

def ogtt(q, t_end=180, dose=None):
    G0, I0 = fasting(q)
    y0 = [q["D"] if dose is None else dose, 0.0, G0, I0, I0]
    t = np.arange(0, t_end + 1.0)
    s = solve_ivp(rhs, (0, t_end), y0, t_eval=t, args=(q,), method="LSODA", rtol=1e-7, atol=1e-9, max_step=2)
    return s.t, s.y

def fluxes(t, y, q):
    """mg/kg(Referenz-KG)/min: Ra, EGP, Muskel-Mehraufnahme; ISR pmol/min/m2"""
    S, Qg, G, I, X = y; c = MW / q["BW"]
    Ra = q["f"] * q["k_abs"] * Qg * 1000 / MW * c
    E = np.array([egp(g, x, q) for g, x in zip(G, X)]) * c
    musc = (q["k_ii_m"] * G + np.array([cl_D(x, q, "m") for x in X]) * G) * c
    musc0 = musc[0]
    return Ra, E, musc - musc0
```

## calibrate.py

Kalibrierung der gesunden Referenz · 48 Zeilen

```python
"""Kalibrierung der gesunden Referenz: Maximum-a-posteriori innerhalb der Literaturstreuung."""
import numpy as np, json, warnings
from scipy.optimize import least_squares
import model as m
warnings.filterwarnings("ignore")
T = np.array([30, 60, 90, 120])
GD = np.array([7.72, 7.26, 6.28, 5.60]); ID = np.array([296, 328, 262, 230.])
FLUX = dict(EGP45=(0.62, 0.3), EGP210=(0.85, 0.3), MUSC=(0.26, 0.06))   # Ferrannini 1985, Kelley 1988
# Prior: Literaturwert, log-SD
PRI = dict(SG=(0.025, .35), SI_D=(9.93e-4, .30), X50_D=(330., .30), X50_L=(71., .40), sP=(0.5, .50), p2=(0.02, .50), CL=(1.6, .20),
           beta=(119., .35), RS=(800., .60), PFR=(2.0, .25), GE_max=(0.5, .20), V_dl=(1.45, .15), k_abs=(0.057, .50))
BOUNDS = dict(GE_max=(0.401, 0.719), PFR=(1.05, 3.5), sP=(0.05, 0.95))
KEYS = list(PRI)

def predict(par, t_end=300):
    q = m.build(par); t, y = m.ogtt(q, t_end)
    Ra, E, mu = m.fluxes(t, y, q)
    musc = mu[:301].sum() * q["BW"] / 1000 / q["D"]          # Anteil der Last (mg/kg/min * kg * min / mg)
    return dict(t=t, G=y[2], I=y[3], EGP=E, Ra=Ra, musc=musc, q=q)

def residuals(z, w_data=1.0):
    par = dict(m.LIT, **{k: float(np.exp(v)) for k, v in zip(KEYS, z)})
    r = predict(par)
    res = list((r["G"][T] - GD) / 0.25 * w_data) + list((r["I"][T] / ID - 1) / 0.12 * w_data)
    res += [(r["EGP"][45] - FLUX["EGP45"][0]) / FLUX["EGP45"][1], (r["EGP"][210] - FLUX["EGP210"][0]) / FLUX["EGP210"][1],
            (r["musc"] - FLUX["MUSC"][0]) / FLUX["MUSC"][1]]
    res += [(v - np.log(PRI[k][0])) / PRI[k][1] for k, v in zip(KEYS, z)]
    res += [max(r["q"]["f"] - 1.0, 0.0) / 0.01]                     # physikalisch: f <= 1
    return np.array(res)

if __name__ == "__main__":
    z0 = np.log([PRI[k][0] for k in KEYS])
    lb = [np.log(BOUNDS.get(k, (PRI[k][0] / 6, 0))[0]) for k in KEYS]
    ub = [np.log(BOUNDS.get(k, (0, PRI[k][0] * 6))[1]) for k in KEYS]
    r0 = residuals(z0); print("Start: Kosten %.1f" % (0.5 * np.sum(r0 ** 2)))
    sol = least_squares(residuals, z0, bounds=(lb, ub), x_scale=1.0, diff_step=1e-3)
    cal = {k: float(np.exp(v)) for k, v in zip(KEYS, sol.x)}
    print("MAP: Kosten %.1f" % sol.cost)
    for k in KEYS:
        print(f"  {k:7s} Literatur {PRI[k][0]:.4g} -> {cal[k]:.4g}  ({(np.log(cal[k]/PRI[k][0]))/PRI[k][1]:+.2f} SD)")
    # Unsicherheit (Laplace)
    J = sol.jac; cov = np.linalg.pinv(J.T @ J); sd = np.sqrt(np.diag(cov))
    par = dict(m.LIT, **cal); r = predict(par)
    print("G", np.round(r["G"][[0, 30, 60, 90, 120]], 2), "Daten 4.95", GD)
    print("I", np.round(r["I"][[0, 30, 60, 90, 120]]), "Daten 38", ID)
    print("EGP45 %.2f EGP210 %.2f Muskel %.2f" % (r["EGP"][45], r["EGP"][210], r["musc"]))
    json.dump(dict(cal=cal, logsd=dict(zip(KEYS, sd.tolist())), cov=cov.tolist(), keys=KEYS, cost=sol.cost, z=sol.x.tolist()), open("calibration.json", "w"), indent=1)
```

## groups.py

Gruppen aus Literaturfaktoren · 66 Zeilen

```python
"""Gruppen aus Literaturfaktoren relativ zur kalibrierten Referenz (keine Anpassung an Gruppendaten)."""
import numpy as np, json, model as m
mg = 1 / 18.016
CAL = dict(m.LIT, **json.load(open("calibration.json"))["cal"])
OBESE = dict(BW=104.0, FM=50.0, FFM=54.0)                     # [L] STEP 1 / Magkos 2016 (FM ~50, FFM ~53-55)

def si_rel(person):
    """Ganzkoerper-Insulinsensitivitaet relativ zur Referenz (Gewichtung Aufnahme/Leber nach Visentin 2015)."""
    P = dict(m.DEFAULT_PERSON, **person); ffm = P["FFM"] / CAL["FFM"]
    D = CAL["musc_SI"] * ffm * P["q_m"] + (1 - CAL["musc_SI"]) * P["q_r"]
    wD = CAL["SI_D"] / (CAL["SI_D"] + CAL["SI_P"])
    return wD * D + (1 - wD) * P["q_L"]

# Umrechnung Clamp-Quotient Rd/I (Mittendorfer 2022: mager 713, adipoes NGT 384, adipoes IGT 216 nmol/kgFFM/min pro mU/L)
# in den Faktor q auf die maximale insulinabh. Clearance. Grund: Rd/I wurde nahe der Saettigung gemessen
# (Clamp-Insulin ~119 mU/L adipoes NGT); die Dosis-Wirkung (X50_D) muss herausgerechnet werden [B].
I_CLAMP_OB = 119 * 6.0                        # [L] pmol/L, WashU-Protokoll 50 mU/m2/min, adipoes NGT
def q_from_rdI(ratio, cl_f, par=None):
    X50 = (par or CAL)["X50_D"]; I_lean = I_CLAMP_OB * 0.77; I_g = I_CLAMP_OB * 0.77 / cl_f   # Clamp-Insulin ~ 1/Clearance
    h = lambda I: I / (X50 + I)
    return (h(I_lean) / I_lean) / ratio * I_g / h(I_g)

def build_groups(par=None, qmult=1.0, beta_t2d=19/50):
    global CAL
    if par is not None: CAL = par
    Q_NGT = q_from_rdI(713 / 384, 0.77) * qmult; Q_IGT = q_from_rdI(713 / 216, 0.68) * qmult
    ob = dict(OBESE, q_m=Q_NGT, q_r=Q_NGT, q_L=0.46, cl_f=0.77, Gb=88*mg, Ib=77.0, PFR=1.6)    # Mittendorfer 2022
    ob["beta_f"] = 1 / si_rel(ob)                                    # hyperbolische Kompensation (Kahn 1993)
    return {
        "gesund": dict(),
        "adipös NGT": ob,
        "adipös IGT": dict(OBESE, q_m=Q_IGT, q_r=Q_IGT, q_L=0.26, cl_f=0.68, Gb=92*mg, Ib=130.0, PFR=1.6,
                           beta_f=ob["beta_f"] * 53/102),            # Ferrannini 2003: IGT 53 vs NGT 102
        "T2D": dict(OBESE, q_m=Q_IGT, q_r=Q_IGT, q_L=0.26, cl_f=0.68, Gb=7.08, Ib=73.0, PFR=1.1, RS_f=0.5,
                    beta_f=ob["beta_f"] * beta_t2d),                 # T2D 19 vs NGT 50 (gleiche Studie)
    }, Q_NGT, Q_IGT
GROUPS, Q_NGT, Q_IGT = build_groups()
_ob_ngt = GROUPS["adipös NGT"]
# Daten (nicht zur Kalibrierung genutzt)
DATA = {
    "gesund": dict(ISR_total=39.4, I_mean120=39400/120/1.6),           # RISC
    "adipös NGT": dict(G120=106*mg, Gmean180=19228/180*mg, Imean180=15245/180*6),   # Mittendorfer 2022
    "adipös IGT": dict(G120=167*mg),
    "T2D": dict(G120=11.85, Imean120=41.2e3/120),
}
def run(group, extra=None, t_end=180, par=None):
    P = dict(GROUPS[group]); P.update(extra or {})
    q = m.build(par or CAL, P); t, y = m.ogtt(q, t_end); return t, y, q

def summary(t, y, q):
    G, I = y[2], y[3]
    P = 1 + (q["PFR"] - 1) * np.minimum(t / 120, 1); dG = np.gradient(G, t)
    ISR = P * np.array([m.isr_basal(g, q) for g in G]) + q["drug_isr"] * q["RS"] * np.maximum(dG, 0)
    return dict(G0=G[0], G30=G[30], G60=G[60], G90=G[90], G120=G[120], Gmean120=G[:121].mean(), Gmean180=G.mean(),
                I0=I[0], I30=I[30], I60=I[60], I120=I[120], Imean120=I[:121].mean(), Imean180=I.mean(),
                ISR_total=np.trapezoid(ISR[:121], t[:121]) / 1000, HOMA=G[0] * I[0] / 6 / 22.5)

if __name__ == "__main__":
    print("q NGT %.3f  q IGT/T2D %.3f  beta_f NGT %.2f" % (Q_NGT, Q_IGT, _ob_ngt["beta_f"]))
    out = {}
    for g in GROUPS:
        s = summary(*run(g)); out[g] = s
        print(f"{g:12s}", " ".join(f"{k}={v:.2f}" if v < 50 else f"{k}={v:.0f}" for k, v in s.items() if k in ("G0","G30","G60","G120","Gmean180","I0","Imean120","Imean180","ISR_total")))
        print(f"{'  Daten':12s}", " ".join(f"{k}={v:.2f}" if v < 50 else f"{k}={v:.0f}" for k, v in DATA[g].items()))
    json.dump(out, open("groups.json", "w"), indent=1)
```

## intervention.py

Fettverlust, Magermasse, GLP-1, Training · 74 Zeilen

```python
"""Interventionen: Gewichtsverlust (Fett/Mager), GLP-1-Direkteffekte, Krafttraining. Nuechternzustand wird neu geloest."""
import numpy as np, model as m, groups as gr

# --- Fettverlust -> Insulinsensitivitaet (Magkos 2016, adipoes NGT, n=9; Clamp bei 5/11/16 % Gewichtsverlust) [L/B]
MK_xFM = np.array([0, 4.9, 9.2, 13.4]) / 50.3
MK_xFFM = np.array([0, 1.5, 2.8, 4.0]) / 53.0
MK_stim = np.array([168, 207, 326, 311]) / 168      # insulinstimulierte Rd-Zunahme relativ
MK_egp = np.array([29, 23, 24, 20]) / 29            # nicht supprimierte EGP im Clamp relativ
Q_FAT = MK_stim / (1 - MK_xFFM)                     # je Muskelmasse [B]
QL_FAT = 1 / MK_egp                                 # Leber [B]
def q_fat(x, s=1.0):  return 1 + s * (float(np.interp(x, MK_xFM, Q_FAT)) - 1)    # ausserhalb: Plateau [A]
def qL_fat(x, s=1.0): return 1 + s * (float(np.interp(x, MK_xFM, QL_FAT)) - 1)

ALPHA = {"gesund": 1.0, "adipös NGT": 1.0, "adipös IGT": 0.5, "T2D": 0.0}   # Betazell-Anpassung [A]
GLP1 = dict(drug_isr=1.45, drug_glucagon=0.86, drug_ge=1.0)                   # Hjerpsted 2018 (adipoes, 12 Wo.)
GLP1_T2D = dict(drug_isr=2.10, drug_glucagon=0.86, drug_ge=1.0)               # Kapitza 2017 (2. Phase x2,1)
STEP1 = dict(xFM=0.193, xLBM=0.097)                                           # Wilding 2021 DXA
RT = dict(gain_FFM=1.0, q_train=1.15)                                         # Metaanalysen ohne Diaet: +1,1 kg (Peterson 2011), +1,5 kg (Benito 2020); Andersen 2003

def make(group, xFM=0.0, xLBM=0.0, gain_FFM=0.0, q_train=1.0, drug=None, alpha=None,
         channels=None, par=None, lean_to_muscle=1.0, s_fat=1.0):
    """Gibt q (Modellparameter) nach Intervention zurueck.
    channels: Teilmenge von {'isr','glucagon','ge','fat_periph','fat_liver','lean','train'} fuer die Zerlegung."""
    par = par or gr.CAL
    base = dict(m.DEFAULT_PERSON, **gr.GROUPS[group]); q0 = m.build(par, base)
    ch = set(channels) if channels is not None else {"isr", "glucagon", "ge", "fat_periph", "fat_liver", "lean", "train"}
    a = ALPHA[group] if alpha is None else alpha
    P = dict(base)
    FFM0 = base["FFM"]; FM0 = base["FM"]
    dFFM = -(xLBM * FFM0) * ("lean" in ch) + gain_FFM * ("train" in ch)
    P["FFM"] = FFM0 + dFFM
    M0 = base.get("M_rel") or FFM0 / par["FFM"]
    P["M_rel"] = M0 * (1 + lean_to_muscle * dFFM / FFM0)          # Anteil der FFM-Aenderung, der Muskel ist [A]
    P["FM"] = FM0 * (1 - xFM); P["BW"] = P["FM"] + FFM0 + dFFM
    if "fat_periph" in ch:
        P["q_m"] = base["q_m"] * q_fat(xFM, s_fat); P["q_r"] = base["q_r"] * q_fat(xFM, s_fat)
    if "train" in ch:
        P["q_m"] = P["q_m"] * q_train
    if "fat_liver" in ch:
        P["q_L"] = base["q_L"] * qL_fat(xFM, s_fat)
    if drug:
        for k in ("isr", "glucagon", "ge"):
            if k in ch: P["drug_" + k] = drug["drug_" + k]
    q = m.build(par, P)
    # Leber-Grundproduktion E0 und Betazell-Lage bleiben Eigenschaften der Person: nicht neu am Anker festsetzen
    r = gr.si_rel(P) / gr.si_rel(base)                         # >1: sensitiver
    adapt = r ** (-a)                                            # hyperbolisch (Kahn 1993), Staerke alpha
    q["E0"] = q0["E0"]
    q["beta"] = q0["beta"] * adapt; q["ISR5"] = q0["ISR5"] * adapt; q["RS"] = q0["RS"] * adapt
    q["glucagon"] = 1 - par["w_glucagon"] * (1 - P["drug_glucagon"])
    return q

def run(q, t_end=180):
    t, y = m.ogtt(q, t_end); return t, y

def metrics(q, t_end=180):
    t, y = run(q, t_end); return gr.summary(t, y, q), t, y

if __name__ == "__main__":
    # ---------- Validierung 1: Magkos 2016 (Gewichtsverlust ohne Medikament, adipoes NGT)
    b, _, _ = metrics(make("adipös NGT"))
    print("Magkos 2016 (5,7/10,9/16,2 % Gewichtsverlust): Modell vs Daten, % Aenderung")
    D = dict(I0=[18.3, 15.5, 12.6, 9.5], Imean120=[12365, 12950, 11137, 9534], Gmean120=[16558, 15790, 17300, 16665])
    for k in D:
        mod = []
        for xf, xl in zip(MK_xFM[1:], MK_xFFM[1:]):
            s, _, _ = metrics(make("adipös NGT", xf, xl)); mod.append(100 * (s[k] / b[k] - 1))
        print(f"  {k:9s} Modell {np.round(mod)}  Daten {np.round(100*(np.array(D[k][1:])/D[k][0]-1))}")
    s, _, _ = metrics(make("adipös NGT", MK_xFM[-1], MK_xFFM[-1])); print("  G0 Modell %+.2f Daten %+.2f mmol/L" % (s["G0"] - b["G0"], (88.6 - 92.7) / 18.016))
    # ---------- Validierung 2: STEP 1 (Semaglutid 2,4 mg, 68 Wo.): Nuechternglukose -8,35 mg/dl
    for g in ["adipös NGT", "adipös IGT"]:
        b, _, _ = metrics(make(g)); s, _, _ = metrics(make(g, **STEP1, drug=GLP1))
        print(f"STEP 1 {g}: dFPG {s['G0']-b['G0']:+.2f} mmol/L (Daten gesamt -0,46) | G120 {b['G120']:.2f} -> {s['G120']:.2f} (normal < 7,8)")
```

## scenarios.py

Szenarien und Shapley-Zerlegung · 79 Zeilen

```python
"""Szenarien, GLP-1-Zerlegung (Shapley) und Unsicherheitsanalyse (Monte Carlo)."""
import numpy as np, json, itertools, math, warnings
import model as m, groups as gr, intervention as iv
warnings.filterwarnings("ignore")
GROUPS_OB = ["adipös NGT", "adipös IGT", "T2D"]
METRICS = ["G0", "G120", "Gmean120", "Imean120", "ISR_total", "I0"]
CH = ["isr", "glucagon", "fat_periph", "fat_liver", "lean"]
CH_LABEL = {"isr": "GLP-1: Betazelle", "glucagon": "GLP-1: Glukagon", "fat_periph": "Fettverlust: Muskel/Peripherie",
            "fat_liver": "Fettverlust: Leber", "lean": "Magermasseverlust"}

def drug_for(g): return iv.GLP1_T2D if g == "T2D" else iv.GLP1
def x_fat_only(g):
    P = gr.GROUPS[g]; return (iv.STEP1["xFM"] * P["FM"] + iv.STEP1["xLBM"] * P["FFM"]) / P["FM"]

def scenario_defs(g, opt=None):
    o = opt or {}; d = drug_for(g); kw = dict(par=o.get("par"), lean_to_muscle=o.get("lm", 1.0), s_fat=o.get("s_fat", 1.0), alpha=o.get("alpha"))
    qt = o.get("q_train", iv.RT["q_train"])
    return {
        "Ausgang": iv.make(g, **kw),
        "GLP-1 (STEP 1)": iv.make(g, **iv.STEP1, drug=d, **kw),
        "GLP-1 + Muskelerhalt": iv.make(g, xFM=iv.STEP1["xFM"], xLBM=iv.STEP1["xLBM"] * 0.5, q_train=qt, drug=d, **kw),
        "GLP-1, nur Fettverlust (theoretisch)": iv.make(g, xFM=x_fat_only(g), xLBM=0.0, drug=d, **kw),
        "Krafttraining ohne GLP-1": iv.make(g, gain_FFM=o.get("gain_FFM", iv.RT["gain_FFM"]), q_train=qt, **kw),
        "Muskel +10 % ohne GLP-1 (hypothetisch)": iv.make(g, gain_FFM=0.10 * gr.GROUPS[g]["FFM"], q_train=qt, **kw),
    }

def shapley(g, metric="Gmean120", opt=None):
    o = opt or {}; d = drug_for(g); kw = dict(par=o.get("par"), lean_to_muscle=o.get("lm", 1.0), s_fat=o.get("s_fat", 1.0), alpha=o.get("alpha"))
    val = {}
    for r in range(len(CH) + 1):
        for S in itertools.combinations(CH, r):
            q = iv.make(g, **iv.STEP1, drug=d, channels=set(S), **kw)
            val[frozenset(S)] = iv.metrics(q)[0][metric]
    n = len(CH); phi = {}
    for c in CH:
        tot = 0.0
        for r in range(n):
            for S in itertools.combinations([x for x in CH if x != c], r):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot += w * (val[frozenset(S) | {c}] - val[frozenset(S)])
        phi[c] = tot
    return phi, val[frozenset()], val[frozenset(CH)]

if __name__ == "__main__":
    out = dict(scen={}, shap={}, curves={}, gast={})
    for g in GROUPS_OB + ["gesund"]:
        if g == "gesund":
            defs = {"Ausgang": iv.make(g), "Krafttraining ohne GLP-1": iv.make(g, gain_FFM=iv.RT["gain_FFM"], q_train=1.15),
                    "Muskel +10 % ohne GLP-1 (hypothetisch)": iv.make(g, gain_FFM=5.2, q_train=1.15)}
        else:
            defs = scenario_defs(g)
        out["scen"][g] = {}; out["curves"][g] = {}
        for k, q in defs.items():
            s, t, y = iv.metrics(q); out["scen"][g][k] = s
            out["curves"][g][k] = dict(G=y[2].tolist(), I=y[3].tolist())
        print(f"\n{g}")
        b = out["scen"][g]["Ausgang"]
        for k, s in out["scen"][g].items():
            print(f"  {k:42s} G0 {s['G0']:.2f} G120 {s['G120']:.2f} ({s['G120']-b['G120']:+.2f})  Gm {s['Gmean120']:.2f} ({s['Gmean120']-b['Gmean120']:+.2f})"
                  f"  Im {s['Imean120']:.0f} ({100*(s['Imean120']/b['Imean120']-1):+.0f} %)  ISR {s['ISR_total']:.0f}")
        if g != "gesund":
            out["shap"][g] = {}
            for met in ["Gmean120", "G120", "Imean120", "G0"]:
                phi, v0, v1 = shapley(g, met); out["shap"][g][met] = dict(phi=phi, base=v0, total=v1)
            ph = out["shap"][g]["Gmean120"]["phi"]
            print("  Zerlegung Gmittel:", {CH_LABEL[k]: round(v, 3) for k, v in ph.items()})
            # Magenentleerung in der fruehen Therapiephase (Hjerpsted 2018: Paracetamol-AUC 0-1 h x0,73)
            q = iv.make(g, **iv.STEP1, drug=dict(drug_for(g), drug_ge=1.0)); s1 = iv.metrics(q)[0]
            # drug_ge so waehlen, dass erste Stunde Darmaufnahme x0,73
            from scipy.optimize import brentq
            def first_hour(fge):
                qq = m.build(gr.CAL, dict(m.DEFAULT_PERSON, drug_ge=fge)); t, y = m.ogtt(qq, 60)
                return np.trapezoid(qq["f"] * qq["k_abs"] * y[1], t)
            ref = first_hour(1.0); fge = brentq(lambda f: first_hour(f) / ref - 0.73, 0.2, 1.0)
            q2 = iv.make(g, **iv.STEP1, drug=dict(drug_for(g), drug_ge=fge)); s2 = iv.metrics(q2)[0]
            out["gast"][g] = dict(fge=fge, dGmean=s2["Gmean120"] - s1["Gmean120"], dG120=s2["G120"] - s1["G120"], dGmax=max(iv.run(q2)[1][2]) - max(iv.run(q)[1][2]))
            print("  Magen (frueh, x%.2f): dGmittel %+.2f dG120 %+.2f dGmax %+.2f" % (fge, out["gast"][g]["dGmean"], out["gast"][g]["dG120"], out["gast"][g]["dGmax"]))
    json.dump(out, open("scenarios.json", "w"))
```

## mc.py

Unsicherheitsanalyse (400 Varianten) · 35 Zeilen

```python
"""Monte-Carlo-Unsicherheit: kalibrierte Parameter (Posterior) + Annahmen [A] gleichzeitig variiert."""
import numpy as np, json, warnings, sys
import model as m, groups as gr, intervention as iv, scenarios as sc
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42); rng_gain = np.random.default_rng(7)   # eigener Strom: uebrige Ziehungen unveraendert
C = json.load(open("calibration.json")); keys = C["keys"]; z = np.array(C["z"]); cov = np.array(C["cov"])
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
base_alpha = dict(iv.ALPHA); base_glp = dict(iv.GLP1); base_t2d = dict(iv.GLP1_T2D)
res = []
for n in range(N):
    zz = rng.multivariate_normal(z, cov)
    par = dict(m.LIT, **{k: float(np.exp(v)) for k, v in zip(keys, zz)})
    par["GE_max"] = float(np.clip(par["GE_max"], 0.401, 0.719)); par["sP"] = float(np.clip(par["sP"], 0.05, 0.95))
    par["musc_SI"] = rng.uniform(0.7, 0.9); par["w_glucagon"] = rng.uniform(0.4, 0.8); par["musc_ii"] = rng.uniform(0.15, 0.45)
    A = dict(s_fat=rng.uniform(0.5, 1.0), lm=rng.uniform(0.5, 1.0), q_train=rng.uniform(1.0, 1.3), qmult=rng.uniform(0.8, 1.2),
             beta_t2d=rng.uniform(0.25, 0.55), a_ngt=rng.uniform(0.5, 1.0), a_igt=rng.uniform(0.0, 1.0), a_t2d=rng.uniform(0.0, 0.5),
             isr=rng.uniform(1.2, 1.7), isr_t2d=rng.uniform(1.6, 2.6), glc=rng.uniform(0.80, 0.92), gain=rng_gain.uniform(0.8, 1.5))
    try:
        gr.CAL = par; gr.GROUPS, _, _ = gr.build_groups(par, A["qmult"], A["beta_t2d"])
        iv.ALPHA = {"gesund": 1.0, "adipös NGT": A["a_ngt"], "adipös IGT": A["a_igt"], "T2D": A["a_t2d"]}
        iv.GLP1 = dict(drug_isr=A["isr"], drug_glucagon=A["glc"], drug_ge=1.0)
        iv.GLP1_T2D = dict(drug_isr=A["isr_t2d"], drug_glucagon=A["glc"], drug_ge=1.0)
        opt = dict(par=par, lm=A["lm"], s_fat=A["s_fat"], q_train=A["q_train"], gain_FFM=A["gain"])
        row = dict(A=A, cal={k: par[k] for k in keys}, scen={}, shap={})
        h = gr.summary(*gr.run("gesund", par=par)); row["healthy"] = h
        for g in sc.GROUPS_OB:
            row["scen"][g] = {k: iv.metrics(q)[0] for k, q in sc.scenario_defs(g, opt).items()}
            phi, v0, v1 = sc.shapley(g, "Gmean120", opt); row["shap"][g] = phi
        res.append(row)
    except Exception as e:
        print("Fehler", n, e)
    if n % 50 == 0: print(n, flush=True)
json.dump(res, open("mc.json", "w"))
print("fertig", len(res))
```

## mc_summary.py

Zusammenfassung der Unsicherheit · 20 Zeilen

```python
import json, numpy as np
R = json.load(open("mc.json")); G = ["adipös NGT", "adipös IGT", "T2D"]
q = lambda a: (np.percentile(a, 50), np.percentile(a, 5), np.percentile(a, 95))
out = {}
for g in G:
    out[g] = {}
    fat = np.array([r["shap"][g]["fat_periph"] + r["shap"][g]["fat_liver"] for r in R])
    lean = np.array([r["shap"][g]["lean"] for r in R]); drug = np.array([r["shap"][g]["isr"] + r["shap"][g]["glucagon"] for r in R])
    out[g]["fat"] = q(fat); out[g]["lean"] = q(lean); out[g]["drug"] = q(drug); out[g]["ratio"] = q(np.abs(fat / lean))
    for k in ["GLP-1 (STEP 1)", "GLP-1 + Muskelerhalt", "GLP-1, nur Fettverlust (theoretisch)", "Krafttraining ohne GLP-1"]:
        d = np.array([r["scen"][g][k]["Gmean120"] - r["scen"][g]["Ausgang"]["Gmean120"] for r in R]); out[g][k] = q(d)
    pot = np.array([r["scen"][g]["GLP-1, nur Fettverlust (theoretisch)"]["Gmean120"] - r["scen"][g]["GLP-1 (STEP 1)"]["Gmean120"] for r in R])
    out[g]["potential"] = q(pot)
    share = np.array([(r["scen"][g]["GLP-1, nur Fettverlust (theoretisch)"]["Gmean120"] - r["scen"][g]["GLP-1 (STEP 1)"]["Gmean120"]) /
                      (r["scen"][g]["GLP-1 (STEP 1)"]["Gmean120"] - r["scen"][g]["Ausgang"]["Gmean120"]) for r in R])
    out[g]["potential_rel"] = q(-100 * share)
    print(g); [print("  %-40s %+.2f [%+.2f; %+.2f]" % (k, *v)) for k, v in out[g].items()]
hg = np.array([[r["healthy"][k] for k in ("G120", "Imean120")] for r in R]); print("gesund G120/Imean120 5-95:", np.percentile(hg, [5, 50, 95], axis=0).round(2))
json.dump(out, open("mc_summary.json", "w"), indent=1)
```

## relevance.py

Parameter-Relevanz ±20 % · 34 Zeilen

```python
import numpy as np, warnings, json, model as m, groups as gr, intervention as iv, scenarios as sc
warnings.filterwarnings("ignore")
CAL0 = dict(gr.CAL)
T = np.array([30,60,90,120]); GD=np.array([7.72,7.26,6.28,5.60]); ID=np.array([296,328,262,230.])
def outputs(par):
    gr.CAL = par; gr.GROUPS, _, _ = gr.build_groups(par)
    t,y,q = gr.run("gesund", par=par)
    fitG = np.sqrt(np.mean((y[2][T]/GD-1)**2))*100; fitI = np.sqrt(np.mean((y[3][T]/ID-1)**2))*100
    res = dict(fitG=fitG, fitI=fitI)
    for g in ["adipös NGT","adipös IGT"]:
        d = sc.drug_for(g)
        b = iv.metrics(iv.make(g, par=par))[0]["Gmean120"]
        f = iv.metrics(iv.make(g, xFM=0.193, par=par))[0]["Gmean120"] - b
        l = iv.metrics(iv.make(g, xLBM=0.097, par=par))[0]["Gmean120"] - b
        res[g+" fett"] = f; res[g+" mager"] = l
    return res
base = outputs(dict(CAL0))
keys = ["SG","SI_D","X50_D","X50_L","sP","p2","CL","beta","RS","PFR","GE_max","V_dl","k_abs","EGP_b","f_ii","U_brain_gd","musc_SI","musc_ii","w_glucagon","GFR","t_half_I"]
rows = []
for k in keys:
    eff = {}
    for fac in (0.8, 1.2):
        p = dict(CAL0); p[k] = CAL0[k]*fac
        if k=="GE_max": p[k]=float(np.clip(p[k],0.401,0.719))
        o = outputs(p); eff[fac] = o
    d = lambda key: max(abs(eff[0.8][key]-base[key]), abs(eff[1.2][key]-base[key]))
    rows.append((k, d("fitG"), d("fitI"), d("adipös IGT fett")/abs(base["adipös IGT fett"])*100, d("adipös IGT mager")/abs(base["adipös IGT mager"])*100, d("adipös NGT mager")/abs(base["adipös NGT mager"])*100))
gr.CAL = CAL0; gr.GROUPS,_,_ = gr.build_groups(CAL0)
print("base", {k: round(v,3) for k,v in base.items()})
print(f"{'Param':12s} dFitG(pp) dFitI(pp) IGT-Fett% IGT-Mager% NGT-Mager%")
for r in sorted(rows, key=lambda r: -max(r[3],r[4])):
    print(f"{r[0]:12s} {r[1]:8.2f} {r[2]:8.2f} {r[3]:9.1f} {r[4]:9.1f} {r[5]:9.1f}")
json.dump(rows, open("relevance.json","w"))
```

## figures.py

Hauptabbildungen · 156 Zeilen

```python
import json, numpy as np, warnings, matplotlib, matplotlib.ticker; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import model as m, groups as gr, intervention as iv, scenarios as sc, calibrate as cb
BLUE, ORANGE, TEAL, PURPLE, GRAY, INK, INK2 = "#2a6fdb", "#c26a17", "#1f9e89", "#8c5bd6", "#9a9994", "#14213D", "#52514e"
GC = {"gesund": GRAY, "adipös NGT": BLUE, "adipös IGT": ORANGE, "T2D": PURPLE}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#ececea",
    "legend.frameon": False, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left"})
S = json.load(open("scenarios.json")); MC = json.load(open("mc_summary.json")); R = json.load(open("mc.json"))
T5 = [0, 30, 60, 90, 120]; GD = [4.95] + list(cb.GD); ID = [38] + list(cb.ID)

# ---------- 1 Gesund: Kalibrierung
r = cb.predict(gr.CAL)
band = np.array([[gr.summary(*gr.run("gesund", par=dict(m.LIT, **x["cal"])))["G120"]] for x in R[:1]])
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
# MC-Band
Gs, Is = [], []
for x in R[:150]:
    t, y, q = gr.run("gesund", par=dict(m.LIT, **x["cal"])); Gs.append(y[2][:181]); Is.append(y[3][:181])
Gs, Is = np.array(Gs), np.array(Is); tt = np.arange(181)
for a, Y, arr, d, err, ttl, yl in [(ax[0], r["G"], Gs, GD, [0] + [0.25]*4, "Glukose", "mmol/L"), (ax[1], r["I"], Is, ID, [0] + list(0.12*np.array(ID[1:])), "Insulin", "pmol/L")]:
    a.fill_between(tt, np.percentile(arr, 5, 0), np.percentile(arr, 95, 0), color=BLUE, alpha=0.15, lw=0, label="90 %-Band (Parameterunsicherheit)")
    a.plot(tt, Y[:181], color=BLUE, lw=2.2, label="Modell (kalibriert)")
    a.errorbar(T5, d, yerr=err, fmt="o", color=INK, ms=6, capsize=3, label="Medianreferenz (4 Kohorten)")
    a.set(title=ttl, xlabel="Zeit nach 75 g Glukose (min)", ylabel=yl); a.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax[1].axhline(39400/120/1.6, color=TEAL, ls="--", lw=1.4, label="RISC: mittleres Insulin 0–120 min")
ax[0].legend(fontsize=9, loc="upper right"); ax[1].legend(fontsize=9, loc="upper right")
t300 = np.arange(301)
ax[2].plot(t300, r["Ra"], color=ORANGE, lw=2, label="Aufnahme aus dem Darm")
ax[2].plot(t300, r["EGP"], color=PURPLE, lw=2, label="Leberproduktion")
ax[2].plot([45, 210], [0.62, 0.85], "s", color=PURPLE, mfc="white", mew=2, ms=8, label="Leber, Tracer (Ferrannini 1985)")
ax[2].plot([210], [2.47], "s", color=ORANGE, mfc="white", mew=2, ms=8, label="Darm, Tracer (Ankerwert)")
ax[2].set(title="Innere Glukoseflüsse", xlabel="Zeit (min)", ylabel="mg/kg/min"); ax[2].legend(fontsize=9, loc="center left", bbox_to_anchor=(0.36, 0.62))
fig.tight_layout(); fig.savefig("fig/f1_gesund.png", dpi=160); plt.close()

# ---------- 2 Gruppen: Vorhersage vs Daten
G = json.load(open("groups.json"))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.9))
for g in gr.GROUPS:
    t, y, q = gr.run(g); ax[0].plot(t, y[2], color=GC[g], lw=2.2, label=g)
dat = {"adipös NGT": [(0, 4.88), (120, 5.88)], "adipös IGT": [(0, 5.11), (120, 9.27)], "T2D": [(0, 7.08), (120, 11.85)], "gesund": list(zip(T5, GD))}
for g, pts in dat.items():
    ax[0].plot(*zip(*pts), "o", color=GC[g], mfc="white", mew=2, ms=8)
ax[0].plot([], [], "o", color=INK, mfc="white", mew=2, label="Studiendaten")
ax[0].axhline(7.8, color=INK2, lw=0.8, ls=":"); ax[0].text(182, 7.8, "IGT-Grenze", va="center", fontsize=9, color=INK2)
ax[0].set(title="Glukose im oGTT je Gruppe", xlabel="Zeit (min)", ylabel="mmol/L"); ax[0].legend(fontsize=9, loc="lower right"); ax[0].set_xlim(0, 200); ax[0].set_ylim(0, 13)
items = [("adipös NGT", "G120", 106/18.016, "2-h-Glukose"), ("adipös NGT", "Gmean180", 19228/180/18.016, "Glukose-Mittel 0–180"),
         ("adipös NGT", "Imean180", 15245/180*6, "Insulin-Mittel 0–180"), ("adipös IGT", "G120", 167/18.016, "2-h-Glukose"),
         ("T2D", "G120", 11.85, "2-h-Glukose"), ("T2D", "Imean120", 41.2e3/120, "Insulin-Mittel 0–120"),
         ("gesund", "ISR_total", 39.4, "Gesamtsekretion (RISC)")]
dev = [100 * (G[g][k] / d - 1) for g, k, d, _ in items]
cols = [GC[g] for g, *_ in items]
ax[1].barh(range(len(items)), dev, color=cols, height=0.6)
for i, v in enumerate(dev): ax[1].text(v + (1 if v >= 0 else -1), i, f"{v:+.0f} %", va="center", ha="left" if v >= 0 else "right", fontsize=10, color=INK)
ax[1].set_yticks(range(len(items)), [f"{g}: {lab}" for g, _, _, lab in items], fontsize=10); ax[1].invert_yaxis()
ax[1].axvline(0, color=INK, lw=0.8); ax[1].axvspan(-10, 10, color=TEAL, alpha=0.08, lw=0)
ax[1].set(title="Abweichung Modell vs. Daten (nicht kalibriert)", xlabel="%"); ax[1].set_xlim(-35, 40); ax[1].grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("fig/f2_gruppen.png", dpi=160); plt.close()

# ---------- 3 Intervention validiert: Magkos + STEP 1
fig, ax = plt.subplots(1, 2, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
b = iv.metrics(iv.make("adipös NGT"))[0]; wl = 100 * (1 - np.array([0, 98.0, 92.9, 87.6]) / 103.8)
wl = np.array([0, 5.7, 10.9, 16.2])
D = dict(I0=[18.3, 15.5, 12.6, 9.5], Imean120=[12365, 12950, 11137, 9534], Gmean120=[16558, 15790, 17300, 16665])
lab = dict(I0="Nüchterninsulin", Imean120="Insulin im oGTT", Gmean120="Glukose im oGTT")
for k, c in zip(D, [BLUE, ORANGE, TEAL]):
    mod = [0] + [100 * (iv.metrics(iv.make("adipös NGT", xf, xl))[0][k] / b[k] - 1) for xf, xl in zip(iv.MK_xFM[1:], iv.MK_xFFM[1:])]
    ax[0].plot(wl, mod, color=c, lw=2.2, label=f"{lab[k]} Modell")
    ax[0].plot(wl, 100 * (np.array(D[k]) / D[k][0] - 1), "o", color=c, mfc="white", mew=2, ms=8, label=f"{lab[k]} Magkos 2016")
ax[0].axhline(0, color=INK, lw=0.8); ax[0].set(title="Gewichtsverlust ohne Medikament (Magkos 2016)", xlabel="Gewichtsverlust (%)", ylabel="Änderung (%)")
ax[0].legend(fontsize=8.5, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
vals = []
for g in ["adipös NGT", "adipös IGT"]:
    b0 = iv.metrics(iv.make(g))[0]; s1 = iv.metrics(iv.make(g, **iv.STEP1, drug=iv.GLP1))[0]; vals.append(s1["G0"] - b0["G0"])
mix = 0.563 * vals[0] + 0.437 * vals[1]
ax[1].bar([0, 1, 2], vals + [mix], color=[BLUE, ORANGE, INK2], width=0.55, label="Modell")
ax[1].bar([3], [-8.35 / 18.016], color=GRAY, width=0.55, label="STEP 1 (Daten)")
for i, v in enumerate(vals + [mix, -8.35 / 18.016]): ax[1].text(i, v - 0.03, f"{v:+.2f}", ha="center", va="top", fontsize=10, color=INK)
ax[1].set_xticks(range(4), ["adipös NGT", "adipös IGT", "Mischung 56/44\n(wie STEP 1)", "STEP 1\ngemessen"])
ax[1].axhline(0, color=INK, lw=0.8); ax[1].set(title="Semaglutid 2,4 mg: Änderung Nüchternglukose", ylabel="mmol/L"); ax[1].set_ylim(-0.8, 0.05)
ax[1].grid(axis="x", visible=False); ax[1].legend(fontsize=9, loc="lower left")
fig.tight_layout(); fig.savefig("fig/f3_interventionen_validiert.png", dpi=160); plt.close()

# ---------- 4 Szenarien mit Unsicherheit
SCN = ["GLP-1 (STEP 1)", "GLP-1 + Muskelerhalt", "GLP-1, nur Fettverlust (theoretisch)", "Krafttraining ohne GLP-1"]
SL = ["GLP-1\n(STEP 1)", "GLP-1 +\nMuskelerhalt", "GLP-1, nur\nFett (theor.)", "Krafttraining\nohne GLP-1"]
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
w = 0.26
for j, g in enumerate(sc.GROUPS_OB):
    for i, k in enumerate(SCN):
        med, lo, hi = MC[g][k]; x = i + (j - 1) * w
        ax[0].bar(x, med, w * 0.92, color=GC[g], label=g if i == 0 else None)
        ax[0].errorbar(x, med, yerr=[[med - lo], [hi - med]], color=INK, capsize=2, lw=1)
    b0 = S["scen"][g]["Ausgang"]
    ax[1].bar(np.arange(4) + (j - 1) * w, [100 * (S["scen"][g][k]["Imean120"] / b0["Imean120"] - 1) for k in SCN], w * 0.92, color=GC[g], label=g)
ax[0].set_xticks(range(4), SL); ax[0].axhline(0, color=INK, lw=0.8)
ax[0].set(title="Mittlere Glukose 0–120 min: Änderung (Median, 90 %-Bereich)", ylabel="mmol/L"); ax[0].legend(fontsize=9, loc="lower right"); ax[0].grid(axis="x", visible=False)
ax[1].set_xticks(range(4), SL); ax[1].axhline(0, color=INK, lw=0.8)
ax[1].set(title="Insulin im oGTT: Änderung (Betazellbelastung)", ylabel="%"); ax[1].legend(fontsize=9, loc="lower right"); ax[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("fig/f4_szenarien.png", dpi=160); plt.close()

# ---------- 5 Zerlegung (Shapley) als Wasserfall
CHC = {"isr": BLUE, "glucagon": TEAL, "fat_periph": ORANGE, "fat_liver": "#e0a35c", "lean": PURPLE}
CHS = {"isr": "Betazelle", "glucagon": "Glukagon", "fat_periph": "Fett→Muskel", "fat_liver": "Fett→Leber", "lean": "Magermasse"}
MECH = {"isr": "GLP-1: Betazelle – cAMP ↑, mehr Insulin je Glukose", "glucagon": "GLP-1: Glukagon ↓ – Leber produziert weniger",
        "fat_periph": "Fettverlust → Muskel: Insulinsignal intakt, mehr GLUT4", "fat_liver": "Fettverlust → Leber: Insulin hemmt Leber besser",
        "lean": "Magermasseverlust: kleinere Glukosesenke, kleineres Volumen"}
from matplotlib.patches import Patch
fig, ax = plt.subplots(1, 3, figsize=(18, 5.8), sharey=False)
for a, g in zip(ax, sc.GROUPS_OB):
    sh = S["shap"][g]["Gmean120"]; cur = sh["base"]
    a.bar(0, cur, color=GRAY, width=0.6); a.text(0, cur + 0.05, f"{cur:.2f}", ha="center", fontsize=10, color=INK)
    for i, c in enumerate(sc.CH, 1):
        v = sh["phi"][c]; a.bar(i, v, bottom=cur, color=CHC[c], width=0.6)
        a.text(i, cur + v + (0.05 if v > 0 else -0.05), f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=10, color=INK)
        cur += v
    a.bar(6, cur, color=INK2, width=0.6); a.text(6, cur + 0.05, f"{cur:.2f}", ha="center", fontsize=10, color=INK)
    a.set_xticks(range(7), ["vorher"] + [CHS[c] for c in sc.CH] + ["mit GLP-1"], rotation=30, ha="right", fontsize=10)
    lo = min(sh["total"], sh["base"]) - 1.2; a.set_ylim(max(lo, 0), sh["base"] + 0.6)
    a.set(title=g, ylabel="mittlere Glukose 0–120 min (mmol/L)"); a.grid(axis="x", visible=False)
fig.legend(handles=[Patch(color=CHC[c], label=MECH[c]) for c in sc.CH], loc="lower center", ncol=3, fontsize=10.5, frameon=False)
fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig("fig/f5_zerlegung.png", dpi=160); plt.close()

# ---------- 6 Potenzial: GLP-1 vs nur Fett
fig, ax = plt.subplots(1, 3, figsize=(18, 4.9))
t = np.arange(181)
for a, g in zip(ax, sc.GROUPS_OB):
    C = S["curves"][g]
    a.plot(t, C["Ausgang"]["G"], color=GRAY, lw=2, label="vor Therapie")
    a.plot(t, C["GLP-1 (STEP 1)"]["G"], color=GC[g], lw=2.4, label="GLP-1 (STEP 1: Fett und Magermasse)")
    a.plot(t, C["GLP-1, nur Fettverlust (theoretisch)"]["G"], color=GC[g], lw=2.4, ls="--", label="gleicher Gewichtsverlust nur Fett")
    a.plot(t, C["Krafttraining ohne GLP-1"]["G"], color=TEAL, lw=2.2, ls=":", label="nur Krafttraining, ohne GLP-1")
    med, lo, hi = MC[g]["potential_rel"]
    a.text(0.98, 0.95, f"ungenutztes Potenzial:\n{-med:.0f} % mehr Senkung\n(90 %: {-hi:.0f}–{-lo:.0f} %)", transform=a.transAxes, ha="right", va="top", fontsize=10, color=INK)
    a.set(title=g, xlabel="Zeit (min)", ylabel="Glukose (mmol/L)"); a.set_xticks([0, 30, 60, 90, 120, 150, 180])
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig/f6_potenzial.png", dpi=160); plt.close()

# ---------- 7 Fett vs Magermasse (Verhaeltnis, MC)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for j, g in enumerate(sc.GROUPS_OB):
    f = MC[g]["fat"]; l = MC[g]["lean"]; d = MC[g]["drug"]
    for i, (v, c) in enumerate([(f, ORANGE), (l, PURPLE), (d, BLUE)]):
        x = j + (i - 1) * 0.26; ax[0].bar(x, v[0], 0.24, color=c, label=["Fettverlust (Muskel+Leber)", "Magermasseverlust", "GLP-1 direkt (Betazelle+Glukagon)"][i] if j == 0 else None)
        ax[0].errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=INK, capsize=2, lw=1)
ax[0].set_xticks(range(3), sc.GROUPS_OB); ax[0].axhline(0, color=INK, lw=0.8); ax[0].grid(axis="x", visible=False)
ax[0].set(title="Beiträge zur Glukoseänderung (Median, 90 %)", ylabel="Δ mittlere Glukose (mmol/L)"); ax[0].legend(fontsize=9, loc="lower left")
data = [[abs((r["shap"][g]["fat_periph"] + r["shap"][g]["fat_liver"]) / r["shap"][g]["lean"]) for r in R] for g in sc.GROUPS_OB]
bp = ax[1].boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.5)
for p, g in zip(bp["boxes"], sc.GROUPS_OB): p.set_facecolor(GC[g]); p.set_alpha(0.35); p.set_edgecolor(GC[g])
for med in bp["medians"]: med.set_color(INK)
ax[1].set_yticks([1, 2, 3], sc.GROUPS_OB); ax[1].set_xscale("log"); ax[1].set_xticks([2, 3, 5, 10, 20], ["2:1", "3:1", "5:1", "10:1", "20:1"]); ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set(title="Verhältnis Fettnutzen : Muskelschaden (400 Modellvarianten)", xlabel="Fett : Magermasse"); ax[1].grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig("fig/f7_fett_vs_muskel.png", dpi=160); plt.close()
print("ok")
```

## fig_pot_ins.py

Potenzial Insulin · 20 Zeilen

```python
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("scenarios.json"))
GRAY, TEAL, INK = "#9a9994", "#1f9e89", "#14213D"
GC = {"adipös NGT": "#2a6fdb", "adipös IGT": "#c26a17", "T2D": "#8c5bd6"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
    "axes.grid": True, "grid.color": "#ececea", "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 22})
fig, ax = plt.subplots(1, 3, figsize=(18, 4.9)); t = np.arange(181)
for a, g in zip(ax, GC):
    C = S["curves"][g]; sc = S["scen"][g]; b = sc["Ausgang"]["Imean120"]
    a.plot(t, C["Ausgang"]["I"], color=GRAY, lw=2, label="vor Therapie")
    a.plot(t, C["GLP-1 (STEP 1)"]["I"], color=GC[g], lw=2.4, label="GLP-1 (STEP 1: Fett und Magermasse)")
    a.plot(t, C["GLP-1, nur Fettverlust (theoretisch)"]["I"], color=GC[g], lw=2.4, ls="--", label="gleicher Gewichtsverlust nur Fett")
    a.plot(t, C["Krafttraining ohne GLP-1"]["I"], color=TEAL, lw=2.2, ls=":", label="nur Krafttraining, ohne GLP-1")
    txt = "Insulin 0–120 min: " + ", ".join(f"{lab} {100*(sc[k]['Imean120']/b-1):+.0f} %" for lab, k in
        [("GLP-1", "GLP-1 (STEP 1)"), ("nur Fett", "GLP-1, nur Fettverlust (theoretisch)"), ("Training", "Krafttraining ohne GLP-1")])
    a.set_title(g, loc="left"); a.text(0, 1.01, txt, transform=a.transAxes, fontsize=10.5, color="#52514e", va="bottom")
    a.set( xlabel="Zeit (min)", ylabel="Insulin (pmol/L)"); a.set_xticks([0, 30, 60, 90, 120, 150, 180]); a.set_ylim(0, None)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=10, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("fig/f6_insulin.png", dpi=160)
```

## fig_valid2.py

Validierung · 35 Zeilen

```python
import numpy as np, warnings, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import intervention as iv
b = iv.metrics(iv.make("adipös NGT"))[0]; s = iv.metrics(iv.make("adipös NGT", iv.MK_xFM[-1], iv.MK_xFFM[-1]))[0]
mod = {k: 100 * (s[k] / b[k] - 1) for k in ("I0", "Gmean120", "Imean120")}
g0 = []; 
for g in ["adipös NGT", "adipös IGT"]:
    b0 = iv.metrics(iv.make(g))[0]; s1 = iv.metrics(iv.make(g, **iv.STEP1, drug=iv.GLP1))[0]; g0.append((b0["G0"], s1["G0"]))
base = 0.563 * g0[0][0] + 0.437 * g0[1][0]; after = 0.563 * g0[0][1] + 0.437 * g0[1][1]
step_mod = 100 * (after / base - 1); step_dat = 100 * (-8.35 / 95.4)
rows = [("Nüchterninsulin", -48, mod["I0"], "passt"),
        ("Glukose im oGTT", 100 * (16665 / 16558 - 1), mod["Gmean120"], "passt (Abweichung < 10 Prozentpunkte)"),
        ("Insulin im oGTT", 100 * (9534 / 12365 - 1), mod["Imean120"], "Modell sinkt etwa doppelt so stark"),
        ("Nüchternglukose", step_dat, step_mod, "passt")]
print([(r[0], round(r[1]), round(r[2])) for r in rows])
INK, INK2, GRAY, BLUE, ORANGE, GREEN = "#14213D", "#52514e", "#a9a8a3", "#2a6fdb", "#c26a17", "#1f9e89"
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
fig, ax = plt.subplots(figsize=(16, 5.6))
y = np.array([0, 1, 2, 3.6]); h = 0.34
for yi, (lab, d, mv, verdict) in zip(y, rows):
    ax.barh(yi - h / 2, d, h, color=GRAY); ax.barh(yi + h / 2, mv, h, color=BLUE)
    for v, yy in [(d, yi - h / 2), (mv, yi + h / 2)]:
        ax.text(v + (-1 if v < 0 else 1), yy, f"{v:+.0f} %", va="center", ha="right" if v < 0 else "left", fontsize=14, color=INK)
    col = ORANGE if "doppelt" in verdict else GREEN
    ax.text(14, yi, verdict, va="center", fontsize=15, color=col, fontweight="bold")
ax.set_yticks(y, [r[0] for r in rows], fontsize=16); ax.invert_yaxis()
ax.axvline(0, color=INK, lw=1); ax.set_xlim(-68, 70); ax.set_xticks([-60, -40, -20, 0])
ax.set_xlabel("Änderung gegenüber vorher (%)", color=INK2); ax.grid(axis="x", color="#ececea"); ax.tick_params(axis="y", length=0)
ax.text(-67, -0.75, "Gewichtsverlust 16 % ohne Medikament (Magkos 2016)", fontsize=15, fontweight="bold", color=INK)
ax.text(-67, 2.85, "Semaglutid 2,4 mg, 68 Wochen (STEP 1)", fontsize=15, fontweight="bold", color=INK)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GRAY, label="gemessen"), Patch(color=BLUE, label="Modell (Vorhersage)")], loc="lower right", fontsize=14, frameon=False, bbox_to_anchor=(1.0, -0.02))
ax.set_ylim(4.1, -1.05)
fig.tight_layout(); fig.savefig("fig/f3b_validierung.png", dpi=160)
```

## fig_glp_groups.py

GLP-1 in vier Gruppen · 32 Zeilen

```python
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import model as m, intervention as iv
GRAY, INK = "#9a9994", "#14213D"
GC = {"gesund": "#1f9e89", "adipös NGT": "#2a6fdb", "adipös IGT": "#c26a17", "T2D": "#8c5bd6"}
NAME = {"gesund": "Gesund", "adipös NGT": "Adipös, normale Glukosetoleranz", "adipös IGT": "Adipös, Prädiabetes (IGT)", "T2D": "Typ-2-Diabetes"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
    "axes.grid": True, "grid.color": "#ececea", "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 22})
fig, ax = plt.subplots(2, 4, figsize=(18, 6.6), sharex=True); t = np.arange(181)
res = {}
for j, g in enumerate(GC):
    drug = iv.GLP1_T2D if g == "T2D" else iv.GLP1
    _, y0 = m.ogtt(iv.make(g), 180); _, y1 = m.ogtt(iv.make(g, **iv.STEP1, drug=drug), 180)
    G0, G1, I0, I1 = y0[2], y1[2], y0[3], y1[3]
    dG = G1[:121].mean() - G0[:121].mean(); dI = 100 * (I1[:121].mean() / I0[:121].mean() - 1)
    res[g] = (G0[0], G1[0], G0[120], G1[120], dG, dI)
    a = ax[0, j]
    a.plot(t, G0, color=GRAY, lw=2, label="vor Therapie"); a.plot(t, G1, color=GC[g], lw=2.6, label="unter GLP-1 (STEP 1)")
    a.axhline(7.8, color="#52514e", lw=1, ls=":"); a.set_ylim(3, 13.5)
    if j == 0: a.text(178, 7.95, "2-h-Grenze IGT", ha="right", fontsize=9.5, color="#52514e")
    a.set_title(NAME[g]); a.text(0, 1.01, f"Glukose Mittel 0–120 min {dG:+.2f} mmol/L".replace(".", ",").replace("-", "−"), transform=a.transAxes, fontsize=10.5, color="#52514e", va="bottom")
    if j == 0: a.set_ylabel("Glukose (mmol/L)")
    b = ax[1, j]
    b.plot(t, I0, color=GRAY, lw=2); b.plot(t, I1, color=GC[g], lw=2.6); b.set_ylim(0, 820)
    b.text(0, 1.01, f"Insulin Mittel 0–120 min {dI:+.0f} %".replace("-", "−"), transform=b.transAxes, fontsize=10.5, color="#52514e", va="bottom")
    b.set_xlabel("Zeit (min)"); b.set_xticks([0, 60, 120, 180])
    if j == 0: b.set_ylabel("Insulin (pmol/L)")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=GRAY, lw=2), Line2D([], [], color=INK, lw=2.6)], ["vor Therapie", "unter Semaglutid 2,4 mg (STEP 1: Fett −19 %, Magermasse −10 %, direkte Wirkungen)"],
           loc="lower center", ncol=2, fontsize=11, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("fig/f8_glp_gruppen.png", dpi=160)
for g, r in res.items(): print(g, " ".join(f"{v:.2f}" for v in r))
```

## fig_base_groups.py

Ausgangslage vier Gruppen · 19 Zeilen

```python
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import model as m, intervention as iv
INK = "#14213D"
GC = {"gesund": "#1f9e89", "adipös NGT": "#2a6fdb", "adipös IGT": "#c26a17", "T2D": "#8c5bd6"}
NAME = {"gesund": "Gesund", "adipös NGT": "Adipös, normal", "adipös IGT": "Adipös, IGT", "T2D": "Typ-2-Diabetes"}
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b4ae",
    "axes.grid": True, "grid.color": "#ececea", "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 12})
fig, ax = plt.subplots(1, 2, figsize=(18, 6.6)); t = np.arange(181)
for g, c in GC.items():
    _, y = m.ogtt(iv.make(g), 180)
    ax[0].plot(t, y[2], color=c, lw=2.8, label=NAME[g]); ax[1].plot(t, y[3], color=c, lw=2.8, label=NAME[g])
    print(g, round(y[2][120],2), round(y[3].max()))
ax[0].axhline(7.8, color="#52514e", lw=1, ls=":"); ax[0].text(100, 7.65, "2-h-Grenze IGT (7,8)", ha="center", va="top", fontsize=11, color="#52514e")
ax[0].set(title="Glukose", ylabel="Glukose (mmol/L)", xlabel="Zeit (min)", ylim=(3, 12.5))
ax[1].set(title="Insulin", ylabel="Insulin (pmol/L)", xlabel="Zeit (min)", ylim=(0, 850))
for a in ax: a.set_xticks([0, 30, 60, 90, 120, 150, 180])
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=13, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("fig/f9_gruppen_basis.png", dpi=160)
```