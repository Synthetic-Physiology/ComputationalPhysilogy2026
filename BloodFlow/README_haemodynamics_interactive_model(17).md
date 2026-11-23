# Hämodynamik im Widerstandsnetzwerk – interaktive Simulation von Aortenbogen, Hirnarterien und Niere

Eine Webseite zu meinem Projekt aus dem Wahlfach Computational Physiology: Das arterielle Gefäßsystem von Kopf, Hals, Aorta und Nieren als Widerstandsnetzwerk. Man stellt eine Stenose oder einen Blutdruck ein und sieht sofort, wie sich Fluss und Druck in jedem Gefäß ändern, welches Hirngebiet unterversorgt ist und was man klinisch erwarten würde.

Die Demo ist eine einzige Datei, [`Interaktive_Demo.html`](Interaktive_Demo.html). Sie braucht keine Installation und kein Internet.

![Kopf und Gehirn: Stenose der A. carotis interna links, 90 %, ohne Verbindungen im Circulus Willisii](abbildungen/demo/demo_gehirn.png)

---

## Schnellstart

| Weg | So geht's |
|---|---|
| **Im Browser, online** | Über GitHub Pages veröffentlichen (siehe [Auf GitHub Pages veröffentlichen](#auf-github-pages-veröffentlichen)) und die Seite unter der Adresse öffnen, die GitHub dort anzeigt. |
| **Lokal** | Repository herunterladen (grüner Knopf *Code* → *Download ZIP*), entpacken, `Interaktive_Demo.html` doppelklicken. Läuft in Chrome, Edge, Firefox und Safari, auch offline. |
| **In Google Colab** | `Seminar_Demo.ipynb` in Colab öffnen und *Alle ausführen* wählen. Das Notebook rechnet die Werte neu (≈ 3–4 min) und zeigt die Seite direkt im Notebook. |

> Wenn man die HTML-Datei auf github.com anklickt, zeigt GitHub nur den Quelltext. Zum Ausprobieren die Datei herunterladen oder über GitHub Pages öffnen.

---

## Bedienung

### Aufbau jeder Seite

Oben stehen vier Reiter: **Grundidee**, **Kopf und Gehirn**, **Niere** und **Quellen**. Die drei Reiter zum Ausprobieren sind gleich aufgebaut:

1. **Überschrift** mit der Frage, um die es geht.
2. **Großes Bild**: Darüber stehen der aktuelle Fall und ein Satz, was sich gegenüber gesund verändert hat.
3. **Steuerleiste**: Schnellwahl, Schalter und Regler. Auf breiten Bildschirmen bleibt sie am unteren Fensterrand stehen, sodass Bild und Regler gleichzeitig zu sehen sind.
4. **Werte**: Tabellen, Kennzahlen und eine Kurve über den ganzen Bereich.

Jede Änderung wirkt sofort, es gibt keinen „Berechnen“-Knopf.

### Die Bilder lesen

| Zeichen | Bedeutung |
|---|---|
| Farbe eines Gefäßes | Druck. Hell bedeutet niedrigen, dunkelblau hohen Druck (im dunklen Modus umgekehrt: heller = höher). Die Skala steht unter dem Bild. |
| Breite eines Gefäßes | Fluss |
| ▲ ▼ vor einem Wert | mehr bzw. weniger Fluss als im Normalzustand |
| „rückwärts“, blau geschrieben | Das Blut fließt gegen die normale Richtung, z. B. in der A. vertebralis beim Subclavian-Steal |
| kleine Pfeile im Gefäß | Flussrichtung in den Verbindungsgefäßen |
| eingefärbtes Hirngebiet | Gelb: vermindert (unter 80 %). Orange: Funktionsausfall (unter 40 %). Rot: Infarktgefahr (unter 20 %). Die Schwellen stammen aus Astrup et al. (1981). |
| schwarzes Schild „Stenose 90 %“ | Ort und Grad der Engstelle |
| gestrichelt | fehlendes Gefäß (Variante) oder hypoplastische A. vertebralis |

**Genaue Werte:** Mit der Maus über ein Gefäß fahren. Auf dem Handy aufs Gefäß tippen; die Werte erscheinen dann unter dem Bild.

### Reiter „Grundidee“

Der Körperkreislauf als Kette von sieben Ebenen: Aorta, Arterien, Arteriolen, Kapillaren, Venolen, Venen und Hohlvenen. Die Ebenen liegen hintereinander, die Gefäße einer Ebene nebeneinander. Hier sieht man, wie das Programm rechnet, bevor es in den anderen Reitern um Kopf und Niere geht.

**Die zwei großen Bilder**

- **Gefäßbaum:** Vom linken Herzen verzweigt sich die Aorta in Arterien und Arteriolen, die Kapillaren bilden ein Netz im Gewebe, Venolen und Venen sammeln das Blut zurück zum rechten Herzen. Die Farbe zeigt den Druck. Darunter stehen, Ebene für Ebene untereinander:
  - der Druckverlauf (gestrichelt zum Vergleich die Ruhe) und der Druckabfall je Ebene;
  - der Gesamtquerschnitt aller Gefäße einer Ebene;
  - die mittlere Geschwindigkeit. Sie ist in den Kapillaren am kleinsten, weil dort der Querschnitt am größten ist.
- **Schaltbild wie in der Elektrotechnik:** Das Herz ist die Spannungsquelle (ΔP = 93 − 5 = 88 mmHg), jede Ebene ein Widerstand, die vielen Gefäße einer Ebene sind als Parallelschaltung gezeichnet. Die Pfeile unter den Widerständen sind die Druckabfälle, wie Spannungspfeile.
  - **Rechenweg:** Ein Klick auf eine Ebene, im Gefäßbaum, im Schaltbild oder in der Tabelle, zeigt ihre Rechnung in fünf Schritten mit den aktuellen Zahlen: Viskosität, Widerstand eines Gefäßes nach Hagen-Poiseuille, Parallelschaltung von N Gefäßen, Druckabfall nach ΔP = Q · R und die ganze Kette.
  - Daneben steht eine Tabelle, die Elektrotechnik und Kreislauf gegenüberstellt.

![Reiter Grundidee: Gefäßbaum mit Druck, Querschnitt und Geschwindigkeit](abbildungen/demo/demo_grundidee.png)

![Schaltbild und Rechenweg für die Arteriolen](abbildungen/demo/demo_schaltbild.png)

**Steuerleiste**

| Bedienelement | Was es macht |
|---|---|
| Radius Arterien | Aorta und Arterien, 50–150 % des Ruhewerts |
| Radius Arteriolen, Kapillaren, Venolen | jeweils diese Ebene, 50–150 % |
| Radius Venen | Venen und Hohlvenen, 50–150 % |
| Hämatokrit | 20–65 %; ändert die Viskosität des Blutes in allen Ebenen |

**Was unter der Steuerleiste steht**

- Herzzeitvolumen, Gesamtwiderstand und Kapillardruck am arteriellen und venösen Ende.
- **Jede Ebene im Überblick:** Anzahl, Durchmesser, Länge, Widerstand mit Anteil, Druckabfall, Druck am Ende und Geschwindigkeit.
- **Wie stark wirkt der Radius jeder Ebene?** Für jede Ebene eine Kurve: der Fluss, wenn man nur ihren Radius verändert. Die Kurve der Arteriolen ist am steilsten.

**Die sechs Fälle der Schnellwahl**

| Schnellwahl | Was passiert |
|---|---|
| Ruhe | 5,0 L/min. Von 88 mmHg Druckgefälle verbrauchen die Arteriolen 53,6 mmHg (61 %), die Kapillaren 20 mmHg (23 %), die Venolen 9,7 mmHg (11 %). Aorta und Arterien zusammen nur 4,4 mmHg (5 %). |
| Arterien halb so weit | Der Fluss sinkt auf 57 %. |
| Arteriolen 20 % enger | Der Fluss sinkt auf 53 %, also so stark wie bei halb so weiten Arterien. Deshalb steuert der Körper seinen Widerstand über die Arteriolen. |
| Arteriolen 20 % weiter | Der Fluss steigt auf 146 %, und der Kapillardruck am arteriellen Ende steigt von 35 auf 49 mmHg. Ein höherer Kapillardruck drückt mehr Flüssigkeit ins Gewebe; so entstehen zum Beispiel die Knöchelödeme unter Kalziumantagonisten vom Dihydropyridin-Typ. |
| Anämie, Hämatokrit 25 % | Die Viskosität fällt von 4,0 auf 2,4 mPa·s, der Fluss steigt auf 166 %. Der Druckverlauf bleibt gleich, weil alle Ebenen im selben Verhältnis leichter durchströmt werden. |
| Polyglobulie, Hämatokrit 60 % | 6,1 mPa·s, der Fluss sinkt auf 66 %. |

Im Modell bleiben 93 mmHg an der Aorta und 5 mmHg in der Hohlvene fest. Im Körper würden Herz und Barorezeptoren gegensteuern; hier sieht man die reine Wirkung der Widerstände.

### Reiter „Kopf und Gehirn“

Das Bild ist eine anatomische Skizze von Kopf, Hals und Gehirn von vorn: Aortenbogen, Karotiden, Vertebralarterien, Circulus Willisii, Großhirn-, Kleinhirn- und Hirnstammarterien. Die rechte Körperseite liegt links im Bild. Neben jedem Gefäß stehen Name, Fluss und Druck.

**Steuerleiste**

| Bedienelement | Was es macht |
|---|---|
| Schnellwahl | sechs fertige Fälle, siehe Tabelle unten |
| Engstelle | *keine*, *Karotis* (A. carotis interna links), *Subclavia* (A. subclavia links, vor dem Abgang der A. vertebralis) oder *Basilaris* |
| Stenosegrad | 0–97 % in 1-%-Schritten; gesperrt, solange keine Engstelle gewählt ist |
| Circulus Willisii | *vollständig*, *ohne ACoA*, *ohne PCoA* (beide) oder *ohne Verbindungen* |
| Vertebralis rechts hypoplastisch | rechte A. vertebralis nur 2 mm weit, eine häufige Normvariante |
| Armarbeit links | Der Widerstand im linken Arm sinkt auf ein Drittel. Nur bei Engstelle *Subclavia* aktiv. |

**Was unter dem Bild steht**

- **Statusschild** oben rechts: das am schlechtesten versorgte Hirngebiet. Sind alle versorgt, steht dort der tiefste Perfusionsdruck.
- **Durchblutung der Hirngebiete:** für jedes Gebiet und jede Seite der Fluss (mL/min und % von gesund) und der Perfusionsdruck. Δ ist die Druckänderung gegenüber gesund. Der rote Strich markiert 60 mmHg; darunter ist die Autoregulation erschöpft. Ein gelber Druckbalken heißt: weniger als 10 mmHg Reserve bis zu dieser Grenze.
- **Was zu erwarten wäre:** Symptome für jedes Gebiet unter 40 %, z. B. Aphasie, Hemianopsie, Wallenberg-Syndrom oder Locked-in-Syndrom (Anatomie 3).
- **Zwei Kurven** über dem ganzen Stenosegrad: oben der Fluss, unten der Perfusionsdruck des schlechtesten Gebiets. Ein Klick in eine Kurve springt zu diesem Grad.

**Die sechs Fälle der Schnellwahl**

| Schnellwahl | Was passiert |
|---|---|
| Normal, ohne Engstelle | Gehirn insgesamt 717 mL/min bei 93 mmHg an der Aortenwurzel. Die Verbindungsgefäße führen in Ruhe kaum Blut. |
| Karotis 90 % | Alle Gebiete bleiben bei 100 %. Die A. communicans anterior springt ein (13 → 197 mL/min). Der Druck in der linken A. cerebri media fällt nur von 85 auf 78 mmHg. |
| … ohne Verbindungen | Dieselbe Stenose ohne Circulus: Mediagebiet und Anteriorgebiet links nur noch 10 %, Infarktgefahr |
| Steal bei Armarbeit | Die linke A. vertebralis fließt rückwärts (285 mL/min), um den Arm zu versorgen. Das Gehirn bleibt trotzdem bei 100 %. Blutdruck am Arm: rechts 92, links 73 mmHg. |
| Steal ohne Umwege | Zusätzlich keine PCoA und eine hypoplastische rechte A. vertebralis: Kleinhirn und Hirnstamm fallen auf etwa 65 %, das Posteriorgebiet auf 60 %. |
| Basilaris 80 %, ohne PCoA | AICA, SCA, Pons und beide Posteriorgebiete bei etwa ein Drittel: Drehschwindel, Hörverlust, Ataxie, Doppelbilder. Die PICA bleibt bei 100 %, weil sie vor der Engstelle abgeht. |

**Warum bleibt die Versorgung oft bei 100 %?**

Mit vollständigem Circulus Willisii kann man den Stenosegrad der Karotis bis 97 % ziehen, und die Tabelle zeigt weiter überall 100 %. Das stimmt so im Modell:

- Durch die Engstelle fließt bei 97 % praktisch nichts mehr (0,03 statt 260 mL/min). Die A. communicans anterior übernimmt (13 → 200 mL/min).
- Der Druck in der linken A. cerebri media fällt dabei nur von 85 auf 78 mmHg. Das sind noch 18 mmHg über der Grenze der Autoregulation (60 mmHg), also halten die Arteriolen den Fluss bei 100 %.
- Klinisch passt das: Ein Verschluss der A. carotis interna bleibt bei gutem Circulus oft ohne Symptome.

Die untere Kurve macht das sichtbar: Der Fluss bleibt flach, der Druck sinkt. Kritisch wird es erst, wenn die Umwege fehlen:

- *Ohne ACoA* liegt der Druck bei 97 % genau an der Grenze von 60 mmHg.
- *Ohne Verbindungen* fällt der Fluss ab etwa 70 %: 93 % bei 70 %, 51 % bei 80 %, 10 % bei 90 %.

![Fluss und Perfusionsdruck bei Karotisstenose mit vollständigem Circulus](abbildungen/demo/demo_gehirn_druck.png)

### Reiter „Niere“

Links ein Körperbild von Modell C (Aorta, Aorta abdominalis, beide Nierenarterien und Nieren), rechts ein einzelnes Nephron mit Vas afferens, Glomerulum und Vas efferens.

| Bedienelement | Was es macht |
|---|---|
| Experiment | *Blutdruck* oder *Nierenarterienstenose* |
| Mitteldruck an der Nierenarterie | 40–200 mmHg (bei *Blutdruck*), mit oder ohne Autoregulation |
| Stenose der A. renalis sinistra | 0–85 % (bei *Nierenarterienstenose*), *mit Angiotensin II*, *unter ACE-Hemmer* oder *ohne Regelung* |
| Umschalter an der Kurve | welche Größe die Kurve zeigt: Glomerulumdruck, GFR, Nierenblutfluss oder Druck hinter der Stenose |

**Ausprobieren**

- *Blutdruck*: den Regler langsam von 150 auf 60 mmHg ziehen. Zwischen 80 und 180 mmHg bleibt die GFR bei 62,5 mL/min, darunter fällt sie. Mit *ohne Autoregulation* stiege der Glomerulumdruck bei 150 mmHg auf 95 mmHg.
- *Nierenarterienstenose*, 65 %, *mit Angiotensin II*: Die linke Niere bekommt 297 statt 550 mL/min, ihre GFR liegt bei 41,6 mL/min. Die rechte Niere bleibt gesund.
- Auf *unter ACE-Hemmer* umschalten: Die GFR fällt auf 2,6 mL/min, obwohl mehr Blut fließt (396 mL/min). Deshalb führen ACE-Hemmer bei beidseitiger Nierenarterienstenose zum Nierenversagen.
- In der Kurve markieren ① und ② die Knicke: ① ist das Vas afferens maximal weit, ② das Vas efferens maximal eng.

![Reiter Niere bei 65 % Stenose der A. renalis](abbildungen/demo/demo_niere.png)

### Reiter „Quellen“

Datengrundlage und Quellen. Oben stehen die Gefäßzahlen der Grundidee: Für jede Ebene Gesamtquerschnitt, Durchmesser, Länge und die daraus gerechnete Anzahl N = A / (π r²), dazu, was aus Guyton und Hall oder Kurbel et al. (2009) stammt, was angenommen und was kalibriert ist. Darunter folgt jeder Modellwert mit seiner Quelle, getrennt nach Grundidee, Kopf und Gehirn, Niere sowie Stenosemodell und Software. Ein Klick auf eine Quelle springt zum Eintrag im Literaturverzeichnis (30 Arbeiten, mit DOI-Links). Eigene Abschnitte nennen die Annahmen ohne Quelle, die kalibrierten Größen, die Entstehung und Nutzung von KI, die empfohlene Zitierweise und den Gültigkeitsbereich.

![Reiter Quellen: Gefäßzahlen der Grundidee mit Herkunft](abbildungen/demo/demo_quellen.png)

### Tipps

- **Tastatur:** Mit Tab zum Regler, dann mit den Pfeiltasten in 1-%- bzw. 1-mmHg-Schritten.
- **Direktlinks:** `Interaktive_Demo.html#grundidee`, `#gehirn`, `#niere` und `#quellen` öffnen direkt den Reiter.
- **Dunkler Modus:** Die Seite folgt der Einstellung des Betriebssystems.
- **Handy und Tablet:** Die Seite passt sich an. Auf schmalen Bildschirmen stehen die Gefäßnamen nicht neben der Skizze, man tippt stattdessen aufs Gefäß.
<img src="abbildungen/demo/demo_handy.png" width="260" alt="Kopf und Gehirn auf dem Handy: Basilarisstenose 80 %, ohne PCoA">

- **Vorführen:** Browser im Vollbild. Ab etwa 1280 Pixeln Breite bleibt die Steuerleiste unten stehen. Eine Ablaufanleitung für 5 Minuten steht in `Interaktive_Demo_Anleitung.txt`.

---

## Was die Demo zeigt

1. **Der Widerstand sitzt in den kleinen Gefäßen.** Nach Hagen-Poiseuille geht der Radius mit der vierten Potenz ein. Im Körperkreislauf der Demo verbrauchen die Arteriolen 61 % des Druckgefälles, die Kapillaren 23 %, Aorta und Arterien nur 5 %. Deshalb hat eine Stenose der großen Arterien lange kaum Wirkung, und deshalb regelt der Körper über die Arteriolen.
2. **Ob eine Engstelle kritisch wird, entscheiden die Umwege.** Mit vollständigem Circulus Willisii bleibt eine 90-%-Karotisstenose folgenlos; ohne Verbindungen führt sie zum Infarkt im Mediagebiet. Auch die Normvarianten (fehlende PCoA, hypoplastische A. vertebralis) bestimmen, ob ein Subclavian-Steal oder eine Basilarisstenose Kleinhirn und Hirnstamm erreicht.
3. **Die Autoregulation hält den Fluss, bis der Druck zu tief fällt.** Großhirn, Kleinhirn und Hirnstamm halten ihren Fluss zwischen 60 und 150 mmHg Perfusionsdruck, die Niere ihre GFR zwischen 80 und 180 mmHg. Hinter einer Nierenarterienstenose hält erst die Weitstellung des Vas afferens und dann Angiotensin II am Vas efferens die Filtration. Ein ACE-Hemmer nimmt diese zweite Stufe weg.

---

## Wie das Programm rechnet

### Vom Python-Modell zur Webseite

```
aortenmodell.py, nierenmodell.py, seminar_demo.py   (Modelle wie im Paper)
        │
        ▼
interaktiv_daten.py         rechnet alle Fälle auf einem feinen Raster (≈ 2 min)
        │
        ▼
interaktiv/modelldaten.json  alle Flüsse und Drücke
        │   + interaktiv/vorlage.html  (Aussehen und Zeichnen)
        ▼
Interaktive_Demo.html        eine Datei, Daten eingebettet, läuft offline
```

Die Seite selbst löst kein Netzwerk. Sie schlägt nur die vorab gerechneten Werte nach und zeichnet sie. Deshalb reagiert sie ohne Verzögerung, und die Regler rasten auf die Rechenpunkte ein. Die Werte sind dieselben wie im Paper und in `seminar_demo.py`. Nur der Reiter „Grundidee“ rechnet live im Browser: Eine Kette von Widerständen in Reihe braucht nur wenige Formeln, und die Seite benutzt dieselben wie `interaktiv_daten.py` (geprüft: gleiche Werte bis auf die letzte Stelle).

### Das Modell

**Gefäße als Widerstände.** Jedes Gefäßsegment ist ein hydraulischer Widerstand nach Hagen-Poiseuille:

$$R = \frac{8\,\mu\,L}{\pi\,r^4}$$

Konische Segmente werden über die Länge integriert. Arteriolen und Kapillaren eines Versorgungsgebiets sind ein peripherer Widerstand. Er ist so kalibriert, dass im gesunden Zustand die physiologischen Flüsse entstehen.

**Netzwerk.** An jedem Knoten gilt Massenerhaltung (1. Kirchhoff-Regel: was hineinfließt, fließt hinaus), entlang jeder Kante ΔP = R · Q, analog zum ohmschen Gesetz. Vorgegeben sind 93 mmHg an der Aortenwurzel und 5 mmHg in den Venen. Unbekannt sind die Drücke aller inneren Knoten. Daraus entsteht ein Gleichungssystem, das ein Newton-Verfahren löst.

**Stenose.** Eine Engstelle ist nach Young und Tsai (1973) ein eigener, nichtlinearer Widerstand:

$$\Delta P = K_v\,\frac{\mu}{D_0}\,U_0 + K_t\,\frac{\rho}{2}\left(\frac{A_0}{A_s}-1\right)^2 U_0\,|U_0|$$

Der erste Teil ist die Reibung, der zweite der Verlust durch Ablösung und Wirbel hinter der Engstelle (K_t = 1,52). Weil der zweite Teil mit dem Quadrat der Verengung und des Flusses wächst, bleibt eine Stenose lange harmlos und wird erst bei hohen Graden plötzlich wirksam.

**Körperkreislauf, Reiter „Grundidee“.** Sieben Ebenen in Reihe, von der Aorta bis zu den Hohlvenen:

- Der Gesamtquerschnitt je Ebene stammt aus Guyton und Hall (Aorta 2,5 cm², Arteriolen 40 cm², Kapillaren 2500 cm², Venolen 250 cm² usw.). Durchmesser und Länge von Arteriolen (30 µm, 1,5 mm) und Kapillaren (5 µm, 0,7 mm) stammen aus Kurbel et al. (2009). Die übrigen Maße sind gerundete Annahmen.
- Die Anzahl je Ebene folgt aus N = Gesamtquerschnitt / (π r²). Die Viskosität hängt vom Hämatokrit ab (Pries et al. 1992, 4,0 mPa·s bei 45 %).
- Drei unsichere Größen werden wie bei der Niere aus Sollwerten der Ruhe bestimmt (Lehrbuchwerte; der venöse Kapillardruck liegt in der Mitte der Angaben von 10–17 mmHg):

| Unsichere Größe | Sollwert | Ergebnis |
|---|---|---|
| Ruhetonus der Arteriolen | Herzzeitvolumen 5 L/min | 26,6 statt 30 µm |
| Anteil durchbluteter Kapillaren | Kapillardruck 35 mmHg am arteriellen Ende | 45 % |
| Zahl der Venolen | Kapillardruck 15 mmHg am venösen Ende | Faktor 0,55 |

Die Anzahlen sind keine Zählungen, sondern folgen aus Querschnitt und Durchmesser: 1 Aorta, 2546 Arterien (bei 1 mm), 5,7 Mio. Arteriolen (bei 30 µm), 12,7 Mrd. Kapillaren (bei 5 µm, davon in Ruhe 5,7 Mrd. durchblutet), 19,5 Mio. Venolen, 2546 Venen (bei 2 mm) und 2 Hohlvenen. Je nach Lehrbuch und Durchmesser liegen solche Zahlen bis zu einer Größenordnung auseinander; Kurbel et al. (2009) kommen mit 4 mm weiten Arterien auf 160 Arterien und auf rund 57 Mio. Arteriolen. Die Ergebnisse hängen daran kaum, weil die Kalibrierung den Widerstand jeder Ebene festlegt: Mit 4 mm weiten Arterien (159 Stück) ändert sich der Ruhetonus der Arteriolen nur von 0,888 auf 0,873; mit zehnmal so vielen Arteriolen stellt sich ihr Ruhedurchmesser auf 15 µm ein, und alle Drücke und Flüsse bleiben gleich. `interaktiv_daten.py` rechnet diese Prüfung jedes Mal mit (Eintrag `pruefung` in `modelldaten.json`).

**Modell B, Kopf und Gehirn.** 42 Gefäße und 18 Versorgungsgebiete: Aortenbogen, Truncus brachiocephalicus, beide Karotiden, Subclavia und Arme, Vertebralarterien, A. basilaris mit Aa. pontis, PICA, AICA und SCA, Circulus Willisii mit ACoA und beiden PCoA, sowie ACA, ACM und PCA beider Seiten.

**Ideale Autoregulation.** Jedes Hirngebiet hält seinen Sollfluss, indem es seinen peripheren Widerstand zwischen einem kleinsten und größten Wert verstellt. Das entspricht einem Perfusionsdruck von 60 bis 150 mmHg. Fällt der Druck unter 60 mmHg, sind die Arteriolen maximal weit, und der Fluss sinkt mit dem Druck.

**Niere.** Die Nierenstrombahn ist eine Kette von der A. renalis bis zur V. renalis über 16 Gefäßebenen mit vielen parallelen Gefäßen je Ebene. Dazu kommen:

- die scheinbare Viskosität nach Pries et al. (1992), also der Fåhræus-Lindqvist-Effekt;
- die glomeruläre Filtration nach Deen, Robertson und Brenner (1972) mit dem onkotischen Druck nach Landis-Pappenheimer;
- die Autoregulation über den Radius des Vas afferens (myogen und tubuloglomeruläres Feedback, ideal zwischen 80 und 180 mmHg);
- Angiotensin II, das bei Bedarf das Vas efferens verengt.

**Modell C, Körperbild.** Modell C ist Modell B plus Aorta abdominalis und beide Nieren. Der Regler setzt den Druck am Abgang der Nierenarterien. Die Werte der Nieren im Körperbild kommen aus derselben Nierenkette wie Nephron und Tabelle.

### Rechenraster

| Teil | Raster | Fälle |
|---|---|---|
| Grundidee | rechnet live | Radius von Arterien, Arteriolen, Kapillaren, Venolen und Venen 50–150 % in 5-%-Schritten, Hämatokrit 20–65 % |
| Kopf und Gehirn | Stenosegrad 0–97 % in 1-%-Schritten | 3 Engstellen × 4 Varianten des Circulus × normale oder hypoplastische A. vertebralis, bei der Subclavia zusätzlich mit und ohne Armarbeit |
| Niere, Blutdruck | 40–200 mmHg in 1-mmHg-Schritten | mit und ohne Autoregulation |
| Niere, Stenose | 0–85 % in 1-%-Schritten | ohne Regelung, unter ACE-Hemmer, mit Angiotensin II |
| Körperbild | wie Niere | dieselben Fälle in Modell C |

### Grenzen

Das Modell ist stationär und nulldimensional. Es gibt keinen Puls, keine Wandelastizität und keine Strömungsprofile. Die Autoregulation ist ideal mit festen Grenzen, und der Blutdruck an der Aortenwurzel bleibt konstant (Barorezeptoren). Die Kette der Grundidee ist eine Lehrkette mit Mittelwerten: Jede Ebene besteht aus gleichen Gefäßen, und die Organe sind nicht einzeln dargestellt. Die Werte zeigen Größenordnungen und Zusammenhänge; für einzelne Patienten sagen sie nichts voraus.

---

## Selbst neu rechnen

```bash
pip install numpy scipy matplotlib
python3 interaktiv_daten.py
```

Das Skript rechnet etwa 2 Minuten und schreibt `interaktiv/modelldaten.json` und eine neue `Interaktive_Demo.html`. Wer nur das Aussehen ändern will, bearbeitet `interaktiv/vorlage.html` und führt das Skript erneut aus.

## Auf GitHub Pages veröffentlichen

1. Projektordner als öffentliches Repository hochladen. `Interaktive_Demo.html` liegt im Hauptordner.
2. Im Repository auf **Settings** gehen, links unter *Code and automation* **Pages** wählen.
3. Unter *Build and deployment* bei *Source* **Deploy from a branch** wählen, dann Branch **main** und Ordner **/ (root)**, und auf **Save** klicken.
4. Nach ein bis zwei Minuten zeigt GitHub unter **Settings → Pages** die Adresse der Seite an. Die Demo liegt unter dieser Adresse mit `/Interaktive_Demo.html` am Ende.

GitHub Pages ist für öffentliche Repositories kostenlos. Die Datei ist etwa 2 MB groß, weil alle Rechenwerte eingebettet sind.

## Quellen, Nutzung von KI und Zitierweise

Jeder Wert, mit dem das Modell rechnet oder kalibriert wird, steht mit Quelle im Reiter „Quellen“. Die Literatur ist die des Papers, ergänzt um Guyton und Hall (2011), Kurbel et al. (2009), Seeley und Young (1976), Landis und Pappenheimer (1963) und Neal et al. (2018); pandas und Matplotlib fehlen, weil die Demo sie zum Rechnen nicht braucht. Autoren, Jahr, Zeitschrift, Band, Seiten, DOI und die übernommenen Zahlenwerte wurden am 23.11.2026 mit Verlagsseiten, PubMed Central und den Abstracts abgeglichen.

**Nutzung von KI.** Beim Programmieren, bei der Literatursuche, bei der Gestaltung der Seite und beim Formulieren der Texte hat der KI-Assistent Claude (Anthropic) mitgearbeitet. Die inhaltliche Verantwortung liegt beim Autor.

**Gültigkeitsbereich.** Lehrmodell auf Grundlage einer mittleren Anatomie, für Studium und Lehre. Die Ergebnisse gelten für keine einzelne Person und ersetzen keine ärztliche Beurteilung.

**Empfohlene Zitierweise.** Winter F. Hämodynamik im Widerstandsnetzwerk: interaktive Simulation von Aortenbogen, Hirnarterien und Niere [Webanwendung]. Wahlfach Computational Physiology; 2026. Datenstand: 23.11.2026.

Auf GitHub Pages ergänzt die Seite im Reiter „Quellen“ die Zitierweise um ihre Adresse und das Abrufdatum.

## Dateien zur Demo

| Datei | Inhalt |
|---|---|
| `Interaktive_Demo.html` | die fertige Demo, eine Datei |
| `interaktiv/vorlage.html` | Aussehen, Zeichnen und Bedienung (HTML, CSS, JavaScript, ohne externe Bibliotheken) |
| `interaktiv/modelldaten.json` | die vorab gerechneten Werte |
| `interaktiv_daten.py` | rechnet die Werte und baut die Seite |
| `Seminar_Demo.ipynb` | Notebook für Colab, rechnet und zeigt die Seite |
| `Interaktive_Demo_Anleitung.txt` | Ablauf für eine 5-minütige Vorführung |
| `Interaktive_Demo_Erklaerung.txt` | kurze Erklärung in einem Absatz pro Reiter |
| `abbildungen/demo/` | Bildschirmfotos für diese Seite |

---

Florian Winter · Wahlfach Computational Physiology · 2026. Ausführliche Herleitung und alle Ergebnisse stehen im Paper und in der [README des Projekts](README.md).
