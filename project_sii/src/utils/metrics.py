"""
src/utils/metrics.py
Modulo Analytics Layer: Calcolo delle correlazioni di Pearson e matrici
per la mappatura del vuoto semantico (Specifiche Punto 2.3).
"""

import os
import json
import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from src.memory.schema import TERMS, DATA_EXTRACTED, REPORTS_DIR

LOG = logging.getLogger("SII_METRICS")

class SemanticsEngine:
    def __init__(self):
        self.extracted_dir = DATA_EXTRACTED
        self.reports_dir = REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)

    def _extract_text_blobs(self, data):
        """Isola i blocchi testuali nidificati all'interno della struttura JSON."""
        text_blocks = []
        if isinstance(data, dict):
            # Cerca paragrafi chiave, didascalie o tabelle estratti dall'InformationExtractor
            for key in ["key_paragraphs", "fig_captions"]:
                if key in data and isinstance(data[key], list):
                    text_blocks += data[key]
            
            # Scansione tabelle/didascalie table
            if "tables" in data and isinstance(data["tables"], list):
                for t in data["tables"]:
                    if isinstance(t, dict) and t.get("caption"):
                        text_blocks.append(t["caption"])
                    elif isinstance(t, str):
                        text_blocks.append(t)
        return text_blocks

    def calculate_correlations(self):
        """
        Carica i documenti estratti, calcola la frequenza semantica locale cumulata,
        applica il coefficiente di Pearson (r) ed esporta i report grafici e tabellari.
        """
        LOG.info("=== Avvio Analisi Semantica: Calcolo Correlazioni di Pearson ===")
        all_counts = []

        if not os.path.exists(self.extracted_dir):
            LOG.warning(f"Directory di input '{self.extracted_dir}' non trovata. Analisi semantica interrotta.")
            return

        # Scansione ricorsiva per mappare i file JSON generati dai 4 corpora indipendenti
        for root, _, files in os.walk(self.extracted_dir):
            for fname in sorted(files):
                if not fname.endswith(".json") or fname == "summary.json":
                    continue
                
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Genera un contatore locale per il file corrente basato sul Thesaurus T
                    counts = Counter()
                    text_blocks = self._extract_text_blobs(data)
                    
                    # Se il JSON conteneva stringhe piatte, le accumuliamo direttamente
                    if not text_blocks and isinstance(data, dict):
                        text_blocks = [str(val) for val in data.values() if isinstance(val, str)]
                    
                    # Calcolo occorrenze
                    for text in text_blocks:
                        low_text = str(text).lower()
                        for cat, terms_list in TERMS.items():
                            for term in terms_list:
                                # Verifica la co-occorrenza del termine isolato
                                counts[cat] += len(re.findall(r'\b' + re.escape(term.lower()) + r'\b', low_text))
                    
                    # Conserviamo l'identificativo relativo del file come indice
                    rel_name = os.path.relpath(path, self.extracted_dir)
                    counts["file"] = rel_name
                    all_counts.append(counts)
                    
                except Exception as e:
                    LOG.error(f"Errore durante l'analisi semantica del file {fname}: {e}")

        if not all_counts:
            LOG.warning("Nessun dato semantico estratto dai file JSON. Impossibile calcolare Pearson.")
            return

        # Costruzione del DataFrame Pandas e riempimento dei vuoti
        df = pd.DataFrame(all_counts).fillna(0).set_index("file")
        
        # Salvataggio delle frequenze assolute
        counts_csv = os.path.join(self.reports_dir, "semantic_counts.csv")
        df.to_csv(counts_csv)
        LOG.info(f"Frequenze semantiche salvate in: {counts_csv}")

        # === Formalizzazione Matematica 3.2: Coefficiente di Pearson (r) ===
        corr = df.corr(method="pearson")
        corr_csv = os.path.join(self.reports_dir, "semantic_correlation.csv")
        corr.to_csv(corr_csv)
        LOG.info(f"Matrice di correlazione di Pearson salvata in: {corr_csv}")

        # === Indice Scalare del Vuoto Semantico ===
        # Una correlazione positiva media vicina a 1 indica convergenza; vicina a 0 indica isolamento.
        gap_rows = []
        off_diagonal_values = []
        columns = list(corr.columns)
        for i, concept_x in enumerate(columns):
            for j, concept_y in enumerate(columns):
                if j <= i:
                    continue
                pearson_r = corr.loc[concept_x, concept_y]
                if pd.isna(pearson_r):
                    continue
                convergence = max(0.0, float(pearson_r))
                pair_gap = round(1.0 - convergence, 4)
                off_diagonal_values.append(convergence)
                gap_rows.append({
                    "concept_x": concept_x,
                    "concept_y": concept_y,
                    "pearson_r": round(float(pearson_r), 4),
                    "pair_semantic_gap": pair_gap
                })

        mean_convergence = sum(off_diagonal_values) / len(off_diagonal_values) if off_diagonal_values else 0.0
        semantic_gap_score = round(1.0 - mean_convergence, 4)
        semantic_convergence_score = round(mean_convergence, 4)

        gap_report = {
            "semantic_gap_score": semantic_gap_score,
            "semantic_convergence_score": semantic_convergence_score,
            "interpretation": "0 = convergenza semantica massima, 1 = vuoto semantico massimo",
            "pairs": gap_rows
        }

        gap_json = os.path.join(self.reports_dir, "semantic_gap_report.json")
        with open(gap_json, "w", encoding="utf-8") as gf:
            json.dump(gap_report, gf, indent=2, ensure_ascii=False)

        gap_csv = os.path.join(self.reports_dir, "semantic_gap_report.csv")
        pd.DataFrame(gap_rows).to_csv(gap_csv, index=False)
        LOG.info(f"Indice scalare del vuoto semantico salvato in: {gap_json}")

        # === Generazione Heatmap Grafica con Annotazioni Cella per Cella ===
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
        plt.colorbar(label="Coefficiente di Correlazione di Pearson (r)")
        
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Mappatura del Vuoto Semantico e Convergenza (TA)")

        # Iniezione dei valori numerici arrotondati all'interno della griglia visiva
        for i in range(len(corr.columns)):
            for j in range(len(corr.index)):
                valore = corr.iloc[j, i]
                # Se non è un NaN, stampa il valore testuale sopra il rispettivo pixel cromatografico
                text_val = f"{valore:.2f}" if not pd.isna(valore) else "N/A"
                plt.text(i, j, text_val, ha='center', va='center', 
                         color='white' if abs(valore) > 0.5 else 'black', fontsize=10, weight='bold')

        plt.tight_layout()
        heatmap_png = os.path.join(self.reports_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_png, dpi=300)
        plt.close()
        LOG.info(f"Heatmap delle correlazioni esportata in: {heatmap_png}")

        # === Grafico Specifico: Vuoto Semantico vs Convergenza ===
        plt.figure(figsize=(7, 4))
        labels = ["Vuoto Semantico", "Convergenza"]
        values = [semantic_gap_score, semantic_convergence_score]
        colors = ["#b91c1c", "#2563eb"]
        bars = plt.bar(labels, values, color=colors)
        plt.ylim(0, 1)
        plt.ylabel("Indice normalizzato (0-1)")
        plt.title("Indice Scalare del Vuoto Semantico (TA)")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.03, 0.98),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                weight="bold"
            )
        plt.tight_layout()
        gap_png = os.path.join(self.reports_dir, "semantic_gap_score.png")
        plt.savefig(gap_png, dpi=300)
        plt.close()
        LOG.info(f"Grafico del vuoto semantico esportato in: {gap_png}")

        print("\n[Analytics] Risultanze Vuoto Semantico")
        print(f"  -> Semantic Gap Score: {semantic_gap_score:.4f} (0=convergenza, 1=vuoto massimo)")
        print(f"  -> Semantic Convergence Score: {semantic_convergence_score:.4f}")
        print(f"  -> Report: {gap_json}")
        print(f"  -> Grafico: {gap_png}")

        return gap_report
