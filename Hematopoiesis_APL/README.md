Agentenbasierte Simulation der Myelopoese, APL (PML-RARα) und ATRA-Differenzierungstherapie

Dieses Projekt modelliert die physiologische Granulopoese sowie die Pathophysiologie und Therapie der **Akuten Promyelozytenleukämie (APL)** mithilfe eines räumlichen agentenbasierten Modells (ABM) in Python.

## Biologischer Hintergrund
* **Physiologische Myelopoese:** 5-stufige Reifungskaskade von der hämatopoetischen Stammzelle (HSC) über GMP, Promyelozyt und Myelozyt bis zum reifen neutrophilen Granulozyten entlang eines räumlichen Knochenmarkgradienten (Endost $\rightarrow$ Sinusoide).
* **Pathophysiologie der APL:** Die Translokation t(15;17) erzeugt das Fusionsprotein **PML-RARα**, das die Ausreifung auf Stufe 3 (**Promyelozyt**) blockiert und zu einer räumlichen Verdrängung der gesunden Hämatopoese (Panzytopenie) führt.
* **ATRA-Therapie & Hidden Progression Clock:** All-trans-Retinsäure (ATRA) hebt den Block transkriptionell auf. Das Modell integriert zwei interne Zell-Uhren (*Hidden Progression Clocks*):
  1. **Maturation Clock:** Bildet die transkriptionelle Latenzphase (*Lag-Phase*) sowie den mitotischen *Point of No Return* ab.
  2. **Escape-Clock:** Zählt Zellteilungen im blockierten Zustand und bildet die Entstehung ATRA-resistenter Subklone (RARα-Ligandenbindungs-Mutation) bei subtherapeutischer Dosierung ab.

## Enthaltenen Skripte & Struktur
* `app.py` – Interaktive Web-Anwendung (Streamlit) zur Live-Exploration verschiedener ATRA-Dosen und Therapiezeitpunkte.
* `healthy_clonal_live.py` – **Nullmodell:** Gesunde Homöostase, räumliche Zonierung und neutraler klonaler Drift über 15 Ursprungs-Stammzellen.
* `apl_atra_cascade.py` – **Mechanistisches Gesamtmodell:** 5-stufige Reifungskaskade, APL-Blockade, transkriptionelle Lag-Phase, Differenzierungssyndrom und Escape-Resistenz.
* `final_evaluation_10x.py` – **Statistische Auswertung:** Reproduzierbarkeitsanalyse über $N = 10$ unabhängige Zufalls-Seeds ($\text{Mean} \pm \text{SD}$), klonales Bottleneck und Dosis-Schwellenwert-Analyse.

## Zentrale Ergebnisse (Statistische Auswertung, N = 10)
![Statistische Gesamtauswertung](auswertung_10_laeufe.png)

1. **Emergente Knochenmarkinsuffizienz:** Allein durch räumliche Platzkonkurrenz sinken die reifen Granulozyten unter APL auf nahe Null.
2. **Differenzierungssyndrom & Lag-Phase:** Nach ATRA-Gabe führt die synchronisierte Ausreifung der aufgestauten Promyelozyten nach einer molekularen Latenzzeit zu einem vorübergehenden Granulozyten-Peak.
3. **Klonales Bottleneck:** Nach erfolgreicher Remission wird das Knochenmark oligoklonal durch wenige überlebende gesunde Stammzell-Linien regeneriert.
4. **Therapeutischer Schwellenwert:** Unterschreitet die ATRA-Reifungsgeschwindigkeit einen kritischen Wert, erreichen einzelne Klone die *Escape-Schwelle* und lösen ein ATRA-resistentes Rezidiv aus (klinische Rationale für die Kombination mit Arsentrioxid, ATO).

## Lokale Ausführung
```bash
pip install -r requirements.txt
python3 apl_atra_cascade.py
