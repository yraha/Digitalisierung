# Anleitung: Anomalieerkennung in Elektromotor-Sensordaten mit einem Autoencoder

Dieses Projekt ist ein lauffähiger Probe-Code für ein Digitalisierungsprojekt
im Bereich Edge AI / Predictive Maintenance. Es erkennt Anomalien im Betrieb
eines Elektromotors (Pumpe/Kompressor, ~15 kW) anhand von Stromverbrauch,
Kugellagertemperatur, Motortemperatur und Drehzahl mithilfe eines Autoencoders
(neuronales Netz).

---

## 1. Projektstruktur

```
projekt_autoencoder_elektromotor/
│
├── data/
│   ├── training/            Gelabelte Trainingsdaten (.xlsx)
│   │   └── trainingsdaten.xlsx
│   └── test/                Hier neue, zu prüfende .xlsx-Dateien ablegen
│       └── testdaten.xlsx   (Demo-Kopie der Trainingsdatei, siehe Abschnitt 5)
│
├── models/                  Trainiertes Modell (wird automatisch erzeugt)
│   ├── autoencoder_model.keras
│   ├── scaler.joblib
│   └── model_metadata.json
│
├── outputs/                 Ergebnisberichte (werden automatisch erzeugt)
│   ├── trainings_bewertung.xlsx
│   └── anomalie_ergebnisse_<Dateiname>.xlsx
│
├── main.py                  Startpunkt (Training + Anomalieerkennung)
├── requirements.txt
└── anleitung.md              Diese Datei
```

---

## 2. Installation (einmalig)

Voraussetzung: Python 3.9 oder neuer.

```bash
cd projekt_autoencoder_elektromotor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Benötigte Bibliotheken

| Bibliothek     | Zweck                                                  | Installationsbefehl        |
|----------------|---------------------------------------------------------|-----------------------------|
| pandas         | Einlesen/Verarbeiten der Excel-Tabellen                  | `pip install pandas`        |
| numpy          | Numerische Berechnungen                                  | `pip install numpy`         |
| openpyxl       | Lese-/Schreib-Engine für .xlsx-Dateien                    | `pip install openpyxl`      |
| scikit-learn   | Skalierung (StandardScaler), Bewertungsmetriken           | `pip install scikit-learn`  |
| joblib         | Speichern/Laden des Scalers                               | `pip install joblib`        |
| tensorflow     | Autoencoder-Modell (Keras)                                 | `pip install tensorflow`    |

Alle Pakete lassen sich auch einzeln nachinstallieren, falls
`requirements.txt` einmal nicht verfügbar ist:

```bash
pip install pandas numpy openpyxl scikit-learn joblib tensorflow
```

---

## 3. Erste Ausführung (Training)

```bash
python3 main.py
```

Beim allerersten Aufruf ist noch kein Modell vorhanden. Das Skript:

1. liest **alle** `.xlsx`-Dateien aus `data/training/` ein,
2. bereinigt die Daten (siehe Abschnitt 6),
3. wählt die als `Normal` gelabelten Zeilen für das Training aus,
4. trainiert den Autoencoder und bestimmt Warn-/Kritisch-Schwellenwerte,
5. speichert Modell, Skalierer (Scaler) und Schwellenwerte in `models/`,
6. bewertet die Erkennungsgüte anhand der vorhandenen Labels und speichert
   den Bericht als `outputs/trainings_bewertung.xlsx`,
7. prüft anschließend automatisch alle Dateien in `data/test/` (siehe
   Abschnitt 4).

Beispielhafte Konsolenausgabe (mit den mitgelieferten Beispieldaten):

```
Accuracy:  95.00%
Precision: 82.61%
Recall:    95.00%
F1-Score:  88.37%
```

Ist bereits ein Modell vorhanden, überspringt `python3 main.py` das Training
und nutzt direkt das gespeicherte Modell (siehe Abschnitt 5). Ein erneutes,
erzwungenes Training (z. B. nach Ergänzung neuer Trainingsdaten) startet man mit:

```bash
python3 main.py --retrain
```

---

## 4. Prüfung neuer Sensordaten (laufender Betrieb)

Sobald ein Modell einmal trainiert wurde, reicht für jede weitere Prüfung:

1. Neue Excel-Datei mit denselben Spalten (siehe Abschnitt 6) in den Ordner
   `data/test/` legen (beliebiger Dateiname, Endung `.xlsx`).
2. Skript ausführen:
   ```bash
   python3 main.py
   ```
3. Ergebnis in `outputs/anomalie_ergebnisse_<Dateiname>.xlsx` öffnen.

Es können auch mehrere neue Dateien gleichzeitig in `data/test/` liegen —
jede Datei wird einzeln geprüft und erhält einen eigenen Ergebnisbericht.
Eine Datei mit fehlerhaftem Spaltenschema wird übersprungen (mit Hinweis in
der Konsole), ohne die Prüfung der übrigen Dateien zu verhindern.

**Hinweis zur mitgelieferten `data/test/testdaten.xlsx`:** Da nur eine
einzige Datenquelle (die Elektromotor-Trainingsdaten) zur Verfügung stand,
enthält `data/test/` aktuell nur eine Kopie dieser Datei. Sie dient allein
dazu, die komplette Pipeline einmal durchlaufen zu lassen (Machbarkeitsnachweis).
Für eine echte Bewertung der Erkennungsgüte auf unabhängigen Daten sollte
diese Datei durch neue, bisher unbekannte Messreihen ersetzt werden.

### Aufbau des Ergebnisberichts

Jeder Bericht (`anomalie_ergebnisse_<Dateiname>.xlsx`) enthält das Tabellenblatt
`Ergebnisse` mit allen Originalspalten plus:

| Neue Spalte              | Bedeutung                                                        |
|---------------------------|--------------------------------------------------------------------|
| `Rekonstruktionsfehler`   | Abweichung (MSE) zwischen Original- und Autoencoder-Rekonstruktion |
| `Anomalie_Status`         | `Normal`, `Warnung` oder `Kritisch`                                 |
| `Ist_Anomalie_erkannt`    | `True`, sobald Status ungleich `Normal`                             |

Enthält die geprüfte Datei zusätzlich eine Label-Spalte (`Hinweis / Zustand`),
werden zwei weitere Blätter (`Kennzahlen`, `Konfusionsmatrix`) mit
Precision/Recall/F1 zur Qualitätskontrolle ergänzt.

---

## 5. Funktionsweise des trainierten Modells bei neuen Daten (Kurzfassung)

1. Die neue Excel-Datei wird eingelesen und genauso bereinigt wie die
   Trainingsdaten (Zeitstempel parsen, unplausible Werte entfernen, kleine
   Messlücken interpolieren).
2. Die vier Merkmale (Stromverbrauch, Kugellagertemperatur,
   Motortemperatur, Drehzahl) werden mit dem beim Training gespeicherten
   `scaler.joblib` skaliert (identische Skalierung wie im Training,
   **nicht** neu angepasst).
3. Das gespeicherte Modell (`autoencoder_model.keras`) versucht, jede Zeile
   zu rekonstruieren. Je größer die Abweichung (Rekonstruktionsfehler),
   desto untypischer ist die Kombination aus Strom, Temperaturen und
   Drehzahl für diesen Motor.
4. Der Fehler wird mit den beim Training bestimmten Schwellenwerten
   (`model_metadata.json`) verglichen und als `Normal`, `Warnung` oder
   `Kritisch` eingestuft.
5. Ergebnis wird als Excel-Bericht in `outputs/` gespeichert.

Es ist **kein erneutes Training** nötig, um neue Daten zu prüfen — das
Modell wird einmal trainiert und danach beliebig oft wiederverwendet.

---

## 6. Erwartetes Excel-Format

Damit neue Dateien korrekt verarbeitet werden, müssen sie (auf dem ersten
oder einem Tabellenblatt) mindestens folgende Spalten enthalten:

| Spaltenname                 | Typ            | Pflicht |
|-------------------------------|-----------------|---------|
| `Zeitstempel`                 | Datum/Uhrzeit    | Ja      |
| `Stromverbrauch [A]`          | Zahl             | Ja      |
| `Kugellagertemperatur [°C]`   | Zahl             | Ja      |
| `Motortemperatur [°C]`        | Zahl             | Ja      |
| `Drehzahl [U/min]`            | Zahl             | Ja      |
| `Betriebsabschnitt`           | Text             | Nein (informativ) |
| `Hinweis / Zustand`           | Text             | Nein (nur für Bewertung) |

Die Spaltennamen müssen exakt übereinstimmen (inkl. Einheiten in eckigen
Klammern). Sollen andere Spaltennamen verwendet werden, müssen sie in
`main.py` in der Klasse `Config` (Felder `timestamp_col`, `feature_cols`,
`label_col`) angepasst werden.

---

## 7. Modellarchitektur (Kurzbeschreibung)

- **Typ:** Vollvernetzter (Dense) Autoencoder, symmetrisch
- **Struktur:** `4 → 12 → 3 → 12 → 4` (Eingabe → Encoder → Bottleneck → Decoder → Ausgabe)
- **Aktivierung:** ReLU in den verdeckten Schichten, lineare Ausgabeschicht
- **Verlustfunktion:** Mean Squared Error (MSE) zwischen Eingabe und Rekonstruktion
- **Optimierer:** Adam, Lernrate 0.001
- **Training:** nur auf als `Normal` gelabelten Datenpunkten, mit Early
  Stopping (Validierungsverlust) gegen Überanpassung
- **Anomalie-Score:** Rekonstruktionsfehler (MSE) je Zeile
- **Schwellenwerte:** 90. Perzentil (Warnung) und 99. Perzentil (Kritisch)
  der Rekonstruktionsfehler auf den normalen Validierungsdaten

Ausführliche Begründungen der Architektur- und Modellentscheidungen befinden
sich als Kommentare direkt im Quellcode (`main.py`), insbesondere in den
Klassen `Config`, `AutoencoderModel` und `AnomalyDetector`.

---

## 8. Bekannte Grenzen dieses Probe-Codes

- Der mitgelieferte Trainingsdatensatz umfasst nur **100 Messpunkte** (rund
  100 Minuten Betrieb) einer einzigen Anlage. Für ein produktives Modell
  sollten deutlich mehr Betriebsstunden über verschiedene Lastzustände und
  Jahreszeiten gesammelt werden.
- Da echte, unabhängige Testdaten bislang fehlen, dient
  `data/test/testdaten.xlsx` nur als Machbarkeitsnachweis (siehe Abschnitt 4).
- Die Schwellenwerte wurden auf Basis dieses kleinen Datensatzes bestimmt
  und sollten bei wachsender Datenbasis regelmäßig durch Neutraining
  (`python3 main.py --retrain`) aktualisiert werden.
