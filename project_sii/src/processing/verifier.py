"""
src/processing/verifier.py
Modulo Processing Layer: Controllo integrità (Auditing), aggregazione statistica 
e generazione di summary strutturati multi-formato (JSON/HTML).
"""

import os
import json
import logging
import statistics
import html as html_lib
from src.memory.schema import DATA_EXTRACTED

LOG = logging.getLogger("SII_VERIFIER")

class PipelineVerifier:
    def __init__(self, extracted_dir=DATA_EXTRACTED):
        self.extracted_dir = extracted_dir

    def generate_summary(self):
        """
        Scansiona ricorsivamente tutti i campioni estratti, aggrega i conteggi globali,
        calcola media e mediana dei valori numerici e compila i report JSON e HTML.
        """
        LOG.info("=== Avvio Modulo Verifier: Generazione Summary e Auditing ===")
        
        if not os.path.exists(self.extracted_dir):
            LOG.error(f"Directory dei dati estratti '{self.extracted_dir}' assente. Impossibile generare il summary.")
            return {"status": "No data available"}

        files_found = []
        total_suspect = 0
        global_term_counts = {}
        file_densities = []
        numeric_counts_list = []

        # Scansione ricorsiva per mappare i 4 corpora indipendenti in sottocartelle
        for root, _, files in os.walk(self.extracted_dir):
            for file in sorted(files):
                if not file.endswith(".json") or file == "summary.json":
                    continue
                
                path = os.path.join(root, file)
                files_found.append(path)
                
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # Recupero sicuro dei dati statistici con fallback
                    stats = data.get("stats", {})
                    suspect = stats.get("suspect_numbers", 0)
                    total_num = stats.get("total_numbers", 0)
                    
                    # Se non presenti chiavi esplicite, cerchiamo in liste di estrazione 'extracted'
                    if total_num == 0 and "extracted" in data:
                        total_num = len(data["extracted"]) if isinstance(data["extracted"], list) else 0
                    
                    total_suspect += suspect
                    numeric_counts_list.append(total_num)
                    
                    # Recupero sicuro metriche semantiche
                    metrics = data.get("semantic_metrics", {})
                    
                    # Se vuote, tentiamo di ricavarle da una scansione piatta o impostiamo dizionario vuoto
                    file_densities.append({
                        "file": os.path.relpath(path, self.extracted_dir),
                        "n_numeric": total_num,
                        "metrics": metrics
                    })
                    
                    # Aggregazione cumulativa delle frequenze per il riepilogo
                    for cat, val in metrics.items():
                        if isinstance(val, (int, float)):
                            global_term_counts[cat] = global_term_counts.get(cat, 0) + val
                            
                except Exception as e:
                    LOG.error(f"Errore durante l'auditing del file {file}: {e}")

        if not files_found:
            LOG.warning("Nessun file JSON valido trovato nelle cartelle di estrazione.")
            return {"status": "No data available"}

        # === Calcolo Metriche di Tendenza Centrale (Cannibalizzato da verifier_ext.py) ===
        mean_numeric = statistics.mean(numeric_counts_list) if numeric_counts_list else 0
        median_numeric = statistics.median(numeric_counts_list) if numeric_counts_list else 0

        # Costruzione del dizionario finale strutturato
        summary = {
            "n_files": len(files_found),
            "n_suspect_values_global": total_suspect,
            "numeric_mean": round(mean_numeric, 2),
            "numeric_median": median_numeric,
            "global_term_counts": global_term_counts,
            "top_dense_files": sorted(file_densities, key=lambda x: x["n_numeric"], reverse=True)[:5]
        }

        # 1. Esportazione Report JSON Standard
        summary_json_path = os.path.join(self.extracted_dir, "summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        LOG.info(f"Report strutturato salvato in: {summary_json_path}")

        # 2. Generazione e compilazione Interfaccia HTML Leggibile (Ispezione Umana)
        try:
            html_rows = []
            for item in file_densities:
                # Elenca le categorie semantiche valorizzate per quel file
                found_cats = [f"{k}({v})" for k, v in item["metrics"].items() if v > 0]
                cats_str = ", ".join(found_cats) if found_cats else "None"
                
                # Sanificazione dei nomi dei file per l'HTML
                safe_file_name = html_lib.escape(item["file"])
                html_rows.append(
                    f"<tr>"
                    f"<td>{safe_file_name}</td>"
                    f"<td>{cats_str}</td>"
                    f"<td>{item['n_numeric']}</td>"
                    f"</tr>"
                )

            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>SII - Extraction & Auditing Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f6f9; color: #333; }}
        h1 {{ color: #1e3a8a; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; display: inline-block; min-width: 250px; margin-right: 20px; }}
        table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #1e3a8a; color: white; }}
        tr:hover {{ background-color: #f1f5f9; }}
    </style>
</head>
<body>
    <h1>Rapporto di Estrazione e Validazione (SII)</h1>
    
    <div class="metric-card">
        <h3>File Elaborati Totali</h3>
        <p style="font-size: 24px; font-weight: bold; color: #1e3a8a;">{summary['n_files']}</p>
    </div>
    <div class="metric-card">
        <h3>Valori Numerici Sospetti</h3>
        <p style="font-size: 24px; font-weight: bold; color: #b91c1c;">{summary['n_suspect_values_global']}</p>
    </div>
    <div class="metric-card">
        <h3>Media / Mediana Campionaria</h3>
        <p style="font-size: 18px; font-weight: bold;">{summary['numeric_mean']} / {summary['numeric_median']}</p>
    </div>

    <h3>Conteggi Globali Categorie Semantiche</h3>
    <ul>
        {"".join([f"<li><b>{html_lib.escape(k)}</b>: {v} occorrenze</li>" for k, v in global_term_counts.items()])}
    </ul>

    <h3>Registro di Dettaglio dei File Normalizzati</h3>
    <table>
        <tr>
            <th>Identificativo File (Sub-Corpus)</th>
            <th>Categorie Rilevate (Frequenza)</th>
            <th># Elementi Numerici</th>
        </tr>
        {"".join(html_rows)}
    </table>
    
    <p style="margin-top: 30px; font-size: 12px; color: #666;">Sistema Automatico SII - Autore: Douglas Ruffini</p>
</body>
</html>
"""
            summary_html_path = os.path.join(self.extracted_dir, "summary.html")
            with open(summary_html_path, "w", encoding="utf-8") as hf:
                hf.write(html_content)
            LOG.info(f"Report visivo umano salvato in: {summary_html_path}")
            
        except Exception as html_err:
            LOG.error(f"Errore durante la compilazione del file HTML: {html_err}")

        return summary