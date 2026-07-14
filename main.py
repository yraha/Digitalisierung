"""
===============================================================================
 Anomalieerkennung in Maschinensensordaten mittels Autoencoder
===============================================================================

Projekt:    projekt_autoencoder_elektromotor
Kontext:    Digitalisierungsprojekt Edge AI / Predictive Maintenance
Maschine:   Industrieller Elektromotor (Pumpe/Kompressor), ~15 kW, Nenndrehzahl 1450 U/min

ZIEL
----
Dieses Skript liest Sensordaten eines Elektromotors aus Excel-Dateien ein,
bereinigt und normiert sie und trainiert einen Autoencoder (neuronales Netz),
der ausschließlich den NORMALEN Betriebszustand der Maschine "lernt".

Die Grundidee der Anomalieerkennung mit Autoencodern:
    Ein Autoencoder komprimiert seine Eingabe auf einen kleinen "Flaschenhals"
    (Bottleneck) und versucht anschließend, daraus die ursprüngliche Eingabe
    wiederherzustellen (Rekonstruktion). Wird das Netz NUR mit Daten aus dem
    Normalbetrieb trainiert, lernt es die typischen Zusammenhänge zwischen
    Stromverbrauch, Kugellagertemperatur, Motortemperatur und Drehzahl sehr
    gut. Weicht ein neuer Messpunkt von diesem gelernten Muster ab (z. B.
    weil eine Lagerüberlastung vorliegt), kann das Netz ihn schlecht
    rekonstruieren -> der Rekonstruktionsfehler (MSE) steigt deutlich an.
    Ein hoher Rekonstruktionsfehler ist somit das Anomalie-Signal.

WARUM AUTOENCODER (statt Klassifikation)?
    - Anomalien sind selten und in der Praxis oft nicht vorab bekannt
      (neue Fehlerbilder, die im Trainingszeitraum noch nicht aufgetreten
      sind). Ein Klassifikator könnte nur die im Training gesehenen
      Fehlerarten erkennen.
    - Der Autoencoder benötigt für das Training nur Normalbetriebsdaten
      und erkennt dadurch auch UNBEKANNTE Abweichungen ("unsupervised
      anomaly detection") - das entspricht der Aufgabenstellung, den
      Zusammenhang zwischen den vier Sensorgrößen zu lernen.

VERWENDUNG DER LABELS ("Hinweis / Zustand")
    Die bereitgestellten Trainingsdaten enthalten eine Spalte mit einer
    manuellen Bewertung (u. a. "Normal" sowie konkrete Anomalie-Texte).
    Diese Labels werden hier NICHT als Haupterkennungsmethode verwendet,
    sondern für zwei unterstützende Zwecke:
      1) Auswahl der "sauberen" Normalbetriebsdaten für das Training
         (Standard-Vorgehen bei Autoencoder-basierter Anomalieerkennung).
      2) Nachträgliche Bewertung der Erkennungsgüte (Precision/Recall/F1),
         da bekannt ist, welche Zeilen tatsächlich Anomalien waren.
    Die eigentliche Erkennung erfolgt ausschließlich über den
    Rekonstruktionsfehler des Autoencoders.

AUSFÜHRUNG
    python main.py                 # trainiert (falls nötig) und prüft data/test/*
    python main.py --retrain       # erzwingt ein erneutes Training

Siehe anleitung.md für eine ausführliche Schritt-für-Schritt-Anleitung.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import psutil
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras erzeugt beim Import viele Infozeilen (CPU-Hinweise etc.),
# die für ein Maschinenbau-Projekt ohne Belang sind - Log-Level reduzieren.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==============================================================================
# 1) KONFIGURATION
# ==============================================================================
# Alle projektspezifischen Einstellungen an einer zentralen Stelle. So kann
# das Skript ohne Codeänderungen an neue Spaltennamen oder Ordnerstrukturen
# angepasst werden.
# ==============================================================================


@dataclass
class Config:
    """Zentrale Projektkonfiguration (Pfade, Spaltennamen, Hyperparameter)."""

    # --- Projektstruktur (relativ zum Speicherort dieses Skripts) -----------
    project_root: Path = Path(__file__).resolve().parent
    training_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)

    # --- Erwartetes Excel-Schema ---------------------------------------------
    # Exakte Spaltennamen wie in der bereitgestellten Datei
    # "Maschinensensordaten_Elektromotor.xlsx" (Blatt "Sensordaten").
    timestamp_col: str = "Zeitstempel"
    label_col: str = "Hinweis / Zustand"
    feature_cols: tuple = (
        "Stromverbrauch [A]",
        "Kugellagertemperatur [°C]",
        "Motortemperatur [°C]",
        "Drehzahl [U/min]",
    )
    normal_label_value: str = "Normal"
    data_sheet_name: str = "Sensordaten"

    # --- Plausibilitätsgrenzen der Sensorik (grobe physikalische Grenzen) ---
    # Werte außerhalb dieser Bereiche gelten als Sensor-/Messfehler und
    # werden in der Datenbereinigung entfernt (keine "Anomalie" im
    # Sinne der Aufgabenstellung, sondern ein defekter Messwert).
    plausible_ranges: dict = field(
        default_factory=lambda: {
            "Stromverbrauch [A]": (0, 100),
            "Kugellagertemperatur [°C]": (-20, 200),
            "Motortemperatur [°C]": (-20, 250),
            "Drehzahl [U/min]": (0, 5000),
        }
    )

    # --- Trainings-Hyperparameter ---------------------------------------------
    validation_split: float = 0.2     # Anteil der Normaldaten für die Validierung
    random_state: int = 42            # Reproduzierbarkeit
    bottleneck_dim: int = 3           # Größe des komprimierten Merkmalsraums
    hidden_dim: int = 12              # Größe der versteckten Schicht
    epochs: int = 300
    batch_size: int = 8               # Kleine Batchgröße wegen kleinem Datensatz
    learning_rate: float = 1e-3
    early_stopping_patience: int = 40

    # --- Schwellenwerte für die Anomalieklassifikation -----------------------
    # Zwei Perzentil-Stufen der Rekonstruktionsfehler auf Normal-
    # Validierungsdaten: "Warnung" und "Kritisch". Perzentile statt
    # fixer Sigma-Grenzen, da bei kleinen Stichproben robuster.
    warning_percentile: float = 90.0
    critical_percentile: float = 99.0

    def __post_init__(self) -> None:
        self.training_dir = self.project_root / "data" / "training"
        self.test_dir = self.project_root / "data" / "test"
        self.models_dir = self.project_root / "models"
        self.outputs_dir = self.project_root / "outputs"
        for d in (self.training_dir, self.test_dir, self.models_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.models_dir / "autoencoder_model.keras"

    @property
    def scaler_path(self) -> Path:
        return self.models_dir / "scaler.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "model_metadata.json"


# ==============================================================================
# 2) DATEN EINLESEN
# ==============================================================================


class ExcelDataLoader:
    """Liest alle Excel-Dateien eines Ordners ein und prüft das Schema."""

    def __init__(self, config: Config):
        self.config = config

    def read_single_file(self, path: Path) -> pd.DataFrame:
        """Liest eine einzelne Excel-Datei ein.

        Es wird zunächst versucht, das Blatt mit dem erwarteten Namen
        ("Sensordaten") zu lesen. Ist dieses nicht vorhanden (z. B. bei
        künftigen Dateien mit abweichendem Blattnamen), wird auf das
        erste Tabellenblatt zurückgegriffen. So bleibt das Skript
        tolerant gegenüber leicht abweichenden neuen Dateien.
        """
        try:
            df = pd.read_excel(path, sheet_name=self.config.data_sheet_name)
        except ValueError:
            df = pd.read_excel(path, sheet_name=0)

        missing = [c for c in self.config.feature_cols if c not in df.columns]
        if self.config.timestamp_col not in df.columns:
            missing.append(self.config.timestamp_col)
        if missing:
            raise ValueError(
                f"Datei '{path.name}' enthält nicht die erwarteten Spalten: {missing}.\n"
                f"Gefundene Spalten: {list(df.columns)}\n"
                f"Erwartet werden mindestens: {[self.config.timestamp_col, *self.config.feature_cols]}"
            )

        df["__quelldatei__"] = path.name
        return df

    def load_folder(self, folder: Path) -> pd.DataFrame:
        """Liest alle .xlsx-Dateien eines Ordners ein und führt sie zusammen."""
        xlsx_files = sorted(folder.glob("*.xlsx"))
        # Temporäre Excel-Sperrdateien (z. B. "~$datei.xlsx") überspringen.
        xlsx_files = [f for f in xlsx_files if not f.name.startswith("~$")]

        if not xlsx_files:
            return pd.DataFrame()

        frames = [self.read_single_file(f) for f in xlsx_files]
        return pd.concat(frames, ignore_index=True)


# ==============================================================================
# 3) DATENBEREINIGUNG UND -AUFBEREITUNG
# ==============================================================================


class DataPreprocessor:
    """Bereinigt Rohdaten und bereitet sie für den Autoencoder auf.

    Schritte:
      1. Zeitstempel parsen, nach Zeit sortieren, exakte Duplikate entfernen.
      2. Merkmalsspalten in numerische Werte umwandeln (fehlerhafte Einträge -> NaN).
      3. Physikalisch unplausible Werte (z. B. negativer Strom) verwerfen.
      4. Kurze Messlücken linear interpolieren, verbleibende NaN-Zeilen entfernen.
    """

    def __init__(self, config: Config):
        self.config = config

    def clean(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        df = df.copy()
        n_start = len(df)

        # 1) Zeitstempel parsen und sortieren -------------------------------
        df[self.config.timestamp_col] = pd.to_datetime(
            df[self.config.timestamp_col], errors="coerce"
        )
        df = df.sort_values(self.config.timestamp_col)
        df = df.drop_duplicates()

        # 2) Merkmale numerisch erzwingen -----------------------------------
        for col in self.config.feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3) Physikalische Plausibilität prüfen ------------------------------
        # Werte außerhalb realistischer Sensorgrenzen werden auf NaN gesetzt
        # und anschließend wie fehlende Werte behandelt (Messfehler-Annahme).
        for col, (lo, hi) in self.config.plausible_ranges.items():
            if col in df.columns:
                out_of_range = ~df[col].between(lo, hi) & df[col].notna()
                if out_of_range.any() and verbose:
                    print(
                        f"  [Bereinigung] {out_of_range.sum()} unplausible Werte in "
                        f"'{col}' außerhalb [{lo}, {hi}] entfernt."
                    )
                df.loc[out_of_range, col] = np.nan

        # 4) Kleine Messlücken interpolieren, Rest verwerfen -------------------
        df[list(self.config.feature_cols)] = df[list(self.config.feature_cols)].interpolate(
            method="linear", limit=3, limit_direction="both"
        )
        n_before_dropna = len(df)
        df = df.dropna(subset=list(self.config.feature_cols))
        n_dropped = n_before_dropna - len(df)
        if n_dropped and verbose:
            print(f"  [Bereinigung] {n_dropped} Zeilen mit fehlenden Messwerten entfernt.")

        df = df.reset_index(drop=True)
        if verbose:
            print(f"  [Bereinigung] {n_start} -> {len(df)} Zeilen nach Bereinigung.")
        return df

    def select_normal_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wählt Zeilen aus, die dem Normalbetrieb zugeordnet sind.

        Begründung: Der Autoencoder soll ausschließlich lernen, wie
        normaler Betrieb (inkl. Anfahr-/Aufwärmvorgängen mit natürlich
        steigenden Temperaturen!) aussieht. Ist keine Label-Spalte
        vorhanden, wird angenommen, dass der gesamte Datensatz den
        Normalbetrieb repräsentiert (Fallback für zukünftige, ungelabelte
        Trainingsdaten).
        """
        if self.config.label_col in df.columns:
            normal = df[df[self.config.label_col] == self.config.normal_label_value]
            return normal.reset_index(drop=True)
        return df.reset_index(drop=True)


# ==============================================================================
# 4) AUTOENCODER-MODELL
# ==============================================================================


class AutoencoderModel:
    """Kapselt Architektur, Training und Rekonstruktion des Autoencoders.

    Architektur (symmetrisch, vollvernetzt):

        Eingabe (4 Merkmale)
            -> Dense(12, ReLU)     Encoder
            -> Dense(3, ReLU)      Bottleneck (komprimierte Repräsentation)
            -> Dense(12, ReLU)     Decoder
            -> Dense(4, linear)    Rekonstruierte Ausgabe (4 Merkmale)

    Begründung der Architekturwahl:
      - Nur 4 Eingangsmerkmale -> ein kleines, flaches Netz reicht aus und
        vermeidet Overfitting auf dem kleinen Trainingsdatensatz.
      - Bottleneck = 3: Die Trainingsdaten umfassen nicht nur den
        stationären Betrieb, sondern auch An- und Abfahrrampen (Start,
        Aufwärmen, Abkühlen), in denen sich alle vier Größen gleichzeitig
        stark ändern. Ein zu enger Flaschenhals (z. B. 2) reicht nicht
        aus, um sowohl stationäre als auch instationäre Normalzustände
        gut abzubilden, und führte in Tests zu Fehlalarmen während
        normaler Anfahr-/Abkühlvorgänge. Bottleneck = 3 bietet genug
        Kapazität für beide Betriebsarten, komprimiert aber weiterhin
        stark genug, dass echte Anomalien (untypische Kombinationen der
        vier Messgrößen) deutlich schlechter rekonstruiert werden.
      - ReLU-Aktivierung in den versteckten Schichten (Standard für Dense-
        Autoencoder), lineare Ausgabeschicht, da die (skalierten)
        Zielwerte auch negativ sein können (StandardScaler).
      - Verlustfunktion: Mean Squared Error (MSE) zwischen Eingabe und
        Rekonstruktion - direkt nutzbar als Rekonstruktionsfehler/
        Anomalie-Score.
      - Adam-Optimierer mit Early Stopping auf den Validierungsverlust,
        um bei nur wenigen Trainingsdaten nicht zu überanpassen.
    """

    def __init__(self, config: Config, input_dim: int):
        self.config = config
        self.input_dim = input_dim
        self.model: Optional[keras.Model] = None

    def build(self) -> keras.Model:
        inputs = keras.Input(shape=(self.input_dim,), name="sensor_features")

        # Encoder
        x = layers.Dense(self.config.hidden_dim, activation="relu", name="encoder_hidden")(inputs)
        bottleneck = layers.Dense(
            self.config.bottleneck_dim, activation="relu", name="bottleneck"
        )(x)

        # Decoder
        x = layers.Dense(self.config.hidden_dim, activation="relu", name="decoder_hidden")(bottleneck)
        outputs = layers.Dense(self.input_dim, activation="linear", name="reconstruction")(x)

        model = keras.Model(inputs, outputs, name="sensor_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
        )
        self.model = model
        return model

    def train(self, x_train: np.ndarray, x_val: np.ndarray) -> keras.callbacks.History:
        """Trainiert den Autoencoder, Eingabe und Ziel sind identisch (x -> x)."""
        assert self.model is not None, "Modell zuerst mit build() erstellen."

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
        )

        history = self.model.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0,
        )
        return history

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Berechnet den Rekonstruktionsfehler (MSE) je Zeile."""
        assert self.model is not None, "Modell nicht geladen."
        reconstructed = self.model.predict(x, verbose=0)
        return np.mean(np.square(x - reconstructed), axis=1)

    def save(self) -> None:
        assert self.model is not None
        self.model.save(self.config.model_path)

    def load(self) -> None:
        self.model = keras.models.load_model(self.config.model_path)


# ==============================================================================
# 5) ANOMALIEERKENNUNG (Schwellenwerte, Klassifikation, Bewertung)
# ==============================================================================


class AnomalyDetector:
    """Bestimmt Schwellenwerte und klassifiziert Rekonstruktionsfehler."""

    def __init__(self, config: Config):
        self.config = config
        self.warning_threshold: Optional[float] = None
        self.critical_threshold: Optional[float] = None

    def fit_thresholds(self, errors_normal: np.ndarray) -> None:
        """Leitet zwei Schwellenwerte aus den Rekonstruktionsfehlern
        des NORMALEN Validierungsdatensatzes ab.

        Perzentil-basiert statt Mittelwert +/- k*Standardabweichung, da bei
        kleinen Stichproben (hier < 100 Werte) robuster gegenüber
        Ausreißern und Verteilungsannahmen.
          - Warnschwelle:   90. Perzentil der Normalfehler
          - Kritischschwelle: 99. Perzentil der Normalfehler
        """
        self.warning_threshold = float(np.percentile(errors_normal, self.config.warning_percentile))
        self.critical_threshold = float(np.percentile(errors_normal, self.config.critical_percentile))
        # Kritischschwelle darf die Warnschwelle nicht unterschreiten
        # (kann bei sehr kleinen/uniformen Stichproben passieren).
        self.critical_threshold = max(self.critical_threshold, self.warning_threshold * 1.0001)

    def classify(self, errors: np.ndarray) -> np.ndarray:
        assert self.warning_threshold is not None, "Schwellenwerte zuerst mit fit_thresholds() bestimmen."
        status = np.full(errors.shape, "Normal", dtype=object)
        status[errors >= self.warning_threshold] = "Warnung"
        status[errors >= self.critical_threshold] = "Kritisch"
        return status

    @staticmethod
    def evaluate_against_labels(status: np.ndarray, labels: pd.Series, normal_value: str) -> dict:
        """Vergleicht die Autoencoder-Erkennung mit vorhandenen Labels.

        Dient AUSSCHLIESSLICH der Bewertung der Erkennungsqualität
        (Precision/Recall/F1/Accuracy), nicht der eigentlichen Erkennung.
        Beide Seiten werden binär betrachtet: "Normal" vs. "Anomalie".
        """
        y_true = (labels != normal_value).astype(int).to_numpy()
        y_pred = (status != "Normal").astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": {
                "true_negative_normal_korrekt": int(cm[0, 0]),
                "false_positive_fehlalarm": int(cm[0, 1]),
                "false_negative_uebersehene_anomalie": int(cm[1, 0]),
                "true_positive_anomalie_korrekt": int(cm[1, 1]),
            },
        }
        return metrics


# ==============================================================================
# 6) TRAININGS-PIPELINE
# ==============================================================================


def run_training(config: Config) -> tuple[AutoencoderModel, AnomalyDetector, StandardScaler]:
    """Führt das komplette Training aus und speichert Modell, Scaler und
    Metadaten (Schwellenwerte, Merkmalsreihenfolge) für die spätere Nutzung.
    """
    print("\n=== TRAINING ===")
    print(f"Lese Trainingsdaten aus: {config.training_dir}")

    loader = ExcelDataLoader(config)
    raw_df = loader.load_folder(config.training_dir)
    if raw_df.empty:
        raise FileNotFoundError(
            f"Keine .xlsx-Dateien in '{config.training_dir}' gefunden. "
            "Bitte mindestens eine gelabelte Trainingsdatei dort ablegen."
        )
    print(f"  {len(raw_df)} Rohzeilen aus {raw_df['__quelldatei__'].nunique()} Datei(en) eingelesen.")

    preprocessor = DataPreprocessor(config)
    clean_df = preprocessor.clean(raw_df)

    # Nur Normalbetriebsdaten für das Training verwenden (siehe Docstring
    # von select_normal_rows für die Begründung).
    normal_df = preprocessor.select_normal_rows(clean_df)
    n_anomal = len(clean_df) - len(normal_df)
    print(
        f"  {len(normal_df)} Zeilen als 'Normalbetrieb' für das Training ausgewählt "
        f"({n_anomal} gelabelte Anomalie-Zeilen werden vom Training ausgeschlossen)."
    )

    if len(normal_df) < 20:
        warnings.warn(
            "Sehr wenige Normalbetriebs-Trainingsdaten (<20 Zeilen). "
            "Das Modell ist ein Probe-/Demonstrationsmodell; für den produktiven "
            "Einsatz sollten deutlich mehr Betriebsstunden gesammelt werden."
        )

    # --- Skalierung ------------------------------------------------------
    # StandardScaler (Mittelwert 0, Standardabweichung 1) wird verwendet,
    # damit alle vier Merkmale trotz unterschiedlicher physikalischer
    # Einheiten (A, °C, °C, U/min) gleich stark zum Rekonstruktionsfehler
    # beitragen. Der Scaler wird NUR auf den Trainingsdaten angepasst.
    feature_cols = list(config.feature_cols)
    x_all_normal = normal_df[feature_cols].to_numpy()

    x_train_raw, x_val_raw = train_test_split(
        x_all_normal,
        test_size=config.validation_split,
        random_state=config.random_state,
        shuffle=True,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)

    print(f"  Trainingssplit: {len(x_train)} Zeilen Training, {len(x_val)} Zeilen Validierung.")

    # --- Autoencoder trainieren -------------------------------------------
    ae = AutoencoderModel(config, input_dim=len(feature_cols))
    ae.build()
    print(f"  Modellarchitektur: {ae.input_dim} -> {config.hidden_dim} -> "
          f"{config.bottleneck_dim} -> {config.hidden_dim} -> {ae.input_dim}")
    history = ae.train(x_train, x_val)
    n_epochs_used = len(history.history["loss"])
    print(f"  Training beendet nach {n_epochs_used} Epochen "
          f"(finaler val_loss = {history.history['val_loss'][-1]:.5f}).")

    # --- Schwellenwerte aus Validierungsfehlern ableiten ---------------------
    detector = AnomalyDetector(config)
    val_errors = ae.reconstruction_error(x_val)
    detector.fit_thresholds(val_errors)
    print(
        f"  Schwellenwerte: Warnung >= {detector.warning_threshold:.5f}, "
        f"Kritisch >= {detector.critical_threshold:.5f} (Rekonstruktions-MSE)."
    )

    # --- Modell, Scaler und Metadaten speichern -------------------------
    ae.save()
    joblib.dump(scaler, config.scaler_path)
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "feature_cols": feature_cols,
        "warning_threshold": detector.warning_threshold,
        "critical_threshold": detector.critical_threshold,
        "n_training_rows_normal": len(x_train),
        "n_validation_rows_normal": len(x_val),
        "epochs_used": n_epochs_used,
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Modell gespeichert: {config.model_path}")
    print(f"  Scaler gespeichert: {config.scaler_path}")
    print(f"  Metadaten gespeichert: {config.metadata_path}")

    # --- Bewertung auf dem GESAMTEN Trainingsdatensatz (inkl. Anomalien) ----
    # Nur zur Qualitätsbewertung - siehe Modulbeschreibung oben.
    if config.label_col in clean_df.columns:
        x_full = scaler.transform(clean_df[feature_cols].to_numpy())
        full_errors = ae.reconstruction_error(x_full)
        status = detector.classify(full_errors)
        metrics = detector.evaluate_against_labels(status, clean_df[config.label_col], config.normal_label_value)

        print("\n  --- Bewertung der Erkennungsqualität anhand der Labels ---")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1-Score:  {metrics['f1_score']:.2%}")
        print(f"  Konfusionsmatrix: {metrics['confusion_matrix']}")

        result_df = clean_df.copy()
        result_df["Rekonstruktionsfehler"] = full_errors
        result_df["Anomalie_Status"] = status
        result_df["Ist_Anomalie_erkannt"] = status != "Normal"
        eval_path = config.outputs_dir / "trainings_bewertung.xlsx"
        with pd.ExcelWriter(eval_path) as writer:
            result_df.to_excel(writer, sheet_name="Bewertung_je_Zeile", index=False)
            pd.DataFrame([metrics["confusion_matrix"]]).to_excel(
                writer, sheet_name="Konfusionsmatrix", index=False
            )
            pd.DataFrame(
                [{k: v for k, v in metrics.items() if k != "confusion_matrix"}]
            ).to_excel(writer, sheet_name="Kennzahlen", index=False)
        print(f"  Bewertungsbericht gespeichert: {eval_path}")

    return ae, detector, scaler


# ==============================================================================
# 7) ERKENNUNGS-PIPELINE FÜR NEUE DATEN
# ==============================================================================


def load_trained_pipeline(config: Config) -> tuple[AutoencoderModel, AnomalyDetector, StandardScaler]:
    """Lädt ein bereits trainiertes Modell samt Scaler und Schwellenwerten."""
    ae = AutoencoderModel(config, input_dim=len(config.feature_cols))
    ae.load()

    scaler: StandardScaler = joblib.load(config.scaler_path)

    metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    detector = AnomalyDetector(config)
    detector.warning_threshold = metadata["warning_threshold"]
    detector.critical_threshold = metadata["critical_threshold"]
    return ae, detector, scaler


def run_detection_on_test_folder(
    config: Config,
    ae: AutoencoderModel,
    detector: AnomalyDetector,
    scaler: StandardScaler,
) -> None:
    """Prüft jede Excel-Datei im Test-Ordner auf Anomalien und schreibt
    je Datei einen Ergebnisbericht in den outputs-Ordner.
    """
    print("\n=== ANOMALIEERKENNUNG AUF NEUEN DATEN ===")
    xlsx_files = sorted(p for p in config.test_dir.glob("*.xlsx") if not p.name.startswith("~$"))

    if not xlsx_files:
        print(
            f"  Keine Dateien in '{config.test_dir}' gefunden.\n"
            "  -> Neue Sensordaten dort als .xlsx ablegen und Skript erneut ausführen."
        )
        return

    preprocessor = DataPreprocessor(config)
    loader = ExcelDataLoader(config)

    for path in xlsx_files:
        print(f"\n  Prüfe Datei: {path.name}")
        try:
            raw_df = loader.read_single_file(path)
        except ValueError as exc:
            # Eine einzelne fehlerhafte Datei (z. B. falsches Spaltenschema)
            # soll die Prüfung der übrigen Dateien nicht verhindern.
            print(f"    -> ÜBERSPRUNGEN: {exc}")
            continue
        clean_df = preprocessor.clean(raw_df, verbose=True)

        x = scaler.transform(clean_df[list(config.feature_cols)].to_numpy())
        errors = ae.reconstruction_error(x)
        status = detector.classify(errors)

        result_df = clean_df.drop(columns=["__quelldatei__"])
        result_df["Rekonstruktionsfehler"] = errors
        result_df["Anomalie_Status"] = status
        result_df["Ist_Anomalie_erkannt"] = status != "Normal"

        n_warn = int(np.sum(status == "Warnung"))
        n_crit = int(np.sum(status == "Kritisch"))
        print(f"    -> {len(result_df)} Zeilen geprüft: {n_warn} Warnung(en), {n_crit} kritische Anomalie(n).")

        sheets = {"Ergebnisse": result_df}

        # Falls (zufällig) auch in den Testdaten Labels vorhanden sind,
        # zusätzlich zur reinen Erkennung eine Bewertung mitliefern.
        if config.label_col in clean_df.columns:
            metrics = detector.evaluate_against_labels(
                status, clean_df[config.label_col], config.normal_label_value
            )
            print(
                f"    -> Bewertung ggü. Label: Precision={metrics['precision']:.2%}, "
                f"Recall={metrics['recall']:.2%}, F1={metrics['f1_score']:.2%}"
            )
            sheets["Kennzahlen"] = pd.DataFrame(
                [{k: v for k, v in metrics.items() if k != "confusion_matrix"}]
            )
            sheets["Konfusionsmatrix"] = pd.DataFrame([metrics["confusion_matrix"]])

        out_path = config.outputs_dir / f"anomalie_ergebnisse_{path.stem}.xlsx"
        with pd.ExcelWriter(out_path) as writer:
            for sheet_name, sheet_df in sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"    -> Ergebnisbericht gespeichert: {out_path}")


# ==============================================================================
# 8) EINSTIEGSPUNKT
# ==============================================================================


def main() -> None:

    ### Rechenleistungsermittlung: Start ###
    start_time = time.time()
    process = psutil.Process(os.getpid())
    ########################################

    parser = argparse.ArgumentParser(
        description="Anomalieerkennung in Elektromotor-Sensordaten mittels Autoencoder."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Erzwingt ein erneutes Training, auch wenn bereits ein Modell gespeichert ist.",
    )
    args = parser.parse_args()

    config = Config()

    # Reproduzierbarkeit: Zufalls-Seeds für NumPy/TensorFlow fixieren.
    np.random.seed(config.random_state)
    tf.random.set_seed(config.random_state)

    model_exists = config.model_path.exists() and config.scaler_path.exists() and config.metadata_path.exists()

    if args.retrain or not model_exists:
        ae, detector, scaler = run_training(config)
    else:
        print("=== Vorhandenes trainiertes Modell wird geladen (kein erneutes Training) ===")
        print(f"  Modell: {config.model_path}")
        ae, detector, scaler = load_trained_pipeline(config)

    run_detection_on_test_folder(config, ae, detector, scaler)

    ### Rechenleistungsermittlung: Ende & Ausgabe ###
    end_time = time.time()
 
    print("Laufzeit:", end_time - start_time, "Sekunden")
    print("RAM-Verbrauch:", process.memory_info().rss / 1024**2, "MB")
    print("CPU-Auslastung:", psutil.cpu_percent(), "%")
    ##################################################

    print("\nFertig.")


if __name__ == "__main__":
    main()
