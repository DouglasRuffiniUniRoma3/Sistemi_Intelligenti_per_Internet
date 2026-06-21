"""
src/processing/extractor.py
Modulo Processing Layer: Parsing lxml, RegEx, calcolo metriche locali,
valutazione degli Anchor Concepts (LLMatch-like) e generazione grafici evolutivi.
"""

import os
import re
import json
import csv
import logging
import matplotlib.pyplot as plt
import pandas as pd
from lxml import html
from src.memory.schema import TERMS, ANCHOR_CONCEPTS, DATA_EXTRACTED, REPORTS_DIR

LOG = logging.getLogger("SII_EXTRACTOR")

class InformationExtractor:
    def __init__(self, source_dir="data/sources/", output_dir=DATA_EXTRACTED, reports_dir=REPORTS_DIR):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.reports_dir = reports_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Pattern numerici e unità di misura fisiche combinate
        self.numeric_unit_pattern = re.compile(
            r"(\d[\d\.\,]*[\s]?(?:[eE××][\+\-]?\d+)?)\s*(V/m|n/cm\^2/s|n/cm2/s|MeV|keV|μs|us|ns|ps|s|ms|kG|pT|T|A|Pa|K|n/s)",
            re.IGNORECASE
        )
        self.generic_numeric_regex = re.compile(r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b')
        self.ta_keywords = r"\b(neutron|gamma|flash|electric|field|discharge|tgf|tge|timing|delay|anomaly|burst)\b"

    def _compute_semantic_metrics(self, text):
        """Calcola le frequenze dei termini del Thesaurus nel testo normalizzato."""
        clean_text = text.lower()
        counts = {category: 0 for category in TERMS}
        for category, keywords in TERMS.items():
            for kw in keywords:
                counts[category] += len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', clean_text))
        return counts

    def _compute_anchor_correlations(self, text):
        """
        Algoritmo LLMatch-like: Calcola il grado di correlazione (0-1) 
        del testo rispetto ai concetti ancoranti della Teoria dell'Avvicinamento.
        """
        clean_text = text.lower()
        scores = {}
        
        for concept, keywords in ANCHOR_CONCEPTS.items():
            matches = 0
            for kw in keywords:
                matches += len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', clean_text))
            
            # Normalizzazione logaritmica Soft-Max per mappare il punteggio nell'intervallo [0, 1]
            # Impedisce che testi molto lunghi superino il valore massimo unitario sbilanciando l'analisi
            if matches > 0:
                score = min(1.0, round((matches / (len(keywords) + matches * 0.2)), 2))
            else:
                score = 0.0
            scores[concept] = score
            
        return scores

    def process_html(self, path, filename):
        """Estrae dati strutturali (tabelle) e testo (paragrafi) usando lxml."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        doc = html.fromstring(text)
        title_tag = doc.xpath("//title/text() | //h1/text()")
        title = title_tag[0].strip() if title_tag else filename
        
        sections_data = []
        full_text_accumulator = title + " "
        explicit_units = []
        
        tables = doc.xpath("//table | //figure") 
        relevant_paras = doc.xpath(
            f"//p[re:match(., '{self.ta_keywords}', 'i') or contains(., 'Table') or contains(., 'Fig.')]", 
            namespaces={'re': 'http://exslt.org/regular-expressions'}
        )
        
        for idx, t in enumerate(tables, start=1):
            caption_text = (t.xpath(".//caption/text() | .//figcaption/text()") or [""])[0].strip()
            table_html = html.tostring(t, encoding='unicode', pretty_print=False)
            numeric_hits = self.numeric_unit_pattern.findall(caption_text)
            
            extracted_vals = [f"{val[0]} {val[1]}" for val in numeric_hits]
            explicit_units.extend(extracted_vals)
            full_text_accumulator += " " + caption_text
            
            sections_data.append({
                "id": f"id_table_{idx}",
                "caption": caption_text,
                "data_type": "table_html",
                "extracted_values": extracted_vals,
                "raw_html": table_html[:200] + "..."
            })
            
        for idx, p in enumerate(relevant_paras):
            ptxt = p.text_content().strip()
            if len(ptxt) > 50:
                numeric_hits = self.numeric_unit_pattern.findall(ptxt)
                extracted_vals = [f"{val[0]} {val[1]}" for val in numeric_hits]
                explicit_units.extend(extracted_vals)
                full_text_accumulator += " " + ptxt
                
                sections_data.append({
                    "id": f"id_paragraph_{idx}",
                    "text": ptxt,
                    "data_type": "text_paragraph",
                    "extracted_values": extracted_vals,
                })

        # =========================================================================
        # PARACADUTE ANTIRUMORE PER AR5IV (FIX DEFINITIVO SUL TEXT MINING)
        # =========================================================================
        if len(full_text_accumulator.strip()) <= len(title) + 5:
            # Cerchiamo prima il contenitore principale del testo del paper di ar5iv
            paper_body = doc.xpath("//article | //div[contains(@class, 'ltx_page_main')] | //div[@id='main']")
            
            if paper_body:
                body_text = paper_body[0].text_content().strip()
            else:
                # Se proprio non lo trova, usa il body ma ripulito
                body_node = doc.xpath("//body")
                body_text = body_node[0].text_content().strip() if body_node else ""
            
            # --- RIMOZIONE SPAZZATURA ACCADEMICA E DI NAVIGAZIONE ---
            # Pulizia via Regex di link di navigazione, stop-words strutturali e rumore di arXiv
            rumore_pattern = r"\b(arxiv|help|pages|classification|search|pdf|abstract|title|authordoc|cookies|terms|feedback|contact|et al|fig|figure|table)\b"
            body_text_clean = re.sub(rumore_pattern, "", body_text, flags=re.IGNORECASE)
            
            # Compatta gli spazi bianchi multipli
            body_text_clean = " ".join(body_text_clean.split())
            full_text_accumulator += " " + body_text_clean
            
            # Popoliamo la sezione per il Verifier
            sections_data.append({
                "id": "id_full_body_clean",
                "text": body_text_clean[:1000] + "...",
                "data_type": "text_paragraph_clean",
                "extracted_values": [f"{val[0]} {val[1]}" for val in self.numeric_unit_pattern.findall(body_text_clean)]
            })
            
            explicit_units.extend([f"{val[0]} {val[1]}" for val in self.numeric_unit_pattern.findall(body_text_clean)])
        # =========================================================================
        
        all_numbers = self.generic_numeric_regex.findall(full_text_accumulator)
        semantic_metrics = self._compute_semantic_metrics(full_text_accumulator)
        anchor_correlations = self._compute_anchor_correlations(full_text_accumulator)
        
        return {
            "source": "arXiv/ar5iv (HTML Engine)",
            "title": title,
            "filename": filename,
            "variables_found": list(set(explicit_units)),
            "semantic_metrics": semantic_metrics,
            "anchor_correlations": anchor_correlations,
            "stats": {
                "total_numbers": len(all_numbers),
                "suspect_numbers": max(0, len(all_numbers) - len(explicit_units))
            },
            "html_sections": sections_data
        }

    def process_csv_txt(self, path, filename):
        """Estrae dati tabellari strutturati o analizza il testo grezzo di fallback."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [f.readline() for _ in range(5)]
            f.seek(0)
            full_body_content = f.read()
            
        content_preview = "".join(first_lines)
        
        if "Title:" in content_preview:
            lines = full_body_content.split("\n")
            title = lines[0].replace("Title: ", "").strip() if lines else "Unknown"
            source = "Local Repository"
            
            for line in lines[:3]:
                if "Source:" in line:
                    source = line.replace("Source: ", "").strip()
            
            explicit_units = [f"{v[0]} {v[1]}" for v in self.numeric_unit_pattern.findall(full_body_content)]
            all_numbers = self.generic_numeric_regex.findall(full_body_content)
            semantic_metrics = self._compute_semantic_metrics(full_body_content)
            anchor_correlations = self._compute_anchor_correlations(full_body_content)
            
            return {
                "source": source,
                "title": title,
                "filename": filename,
                "variables_found": explicit_units,
                "semantic_metrics": semantic_metrics,
                "anchor_correlations": anchor_correlations,
                "stats": {
                    "total_numbers": len(all_numbers),
                    "suspect_numbers": max(0, len(all_numbers) - len(explicit_units))
                },
                "raw_text_preview": full_body_content[:400] + "..."
            }
            
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                try:
                    dialect = csv.Sniffer().sniff(content[:2048], delimiters=',;')
                except csv.Error:
                    dialect = 'excel'
                
                f.seek(0)
                reader = csv.reader(f, dialect)
                header = next(reader)
                
                preview_rows = []
                for i, row in enumerate(reader):
                    if i < 10:
                        preview_rows.append(row)
                    else:
                        break
                
                analyzed_variables = []
                combined_headers_text = " ".join(header)
                
                for h in header:
                    match = re.search(r'\(([^)]+)\)', h)
                    unit = match.group(1).strip() if match else ""
                    is_ta_relevant = bool(re.search(self.ta_keywords, h, re.IGNORECASE))
                    analyzed_variables.append({"name": h, "unit": unit, "is_ta_relevant": is_ta_relevant})
            
            sample_text = combined_headers_text + " " + " ".join([" ".join(r) for r in preview_rows])
            explicit_units = [f"{v[0]} {v[1]}" for v in self.numeric_unit_pattern.findall(sample_text)]
            all_numbers = self.generic_numeric_regex.findall(sample_text)
            semantic_metrics = self._compute_semantic_metrics(sample_text)
            anchor_correlations = self._compute_anchor_correlations(sample_text)
            
            return {
                "source": "Zenodo/Figshare Dataset (CSV Engine)",
                "title": f"Dataset Parameters: {filename}",
                "filename": filename,
                "variables_found": explicit_units,
                "semantic_metrics": semantic_metrics,
                "anchor_correlations": anchor_correlations,
                "stats": {
                    "total_numbers": len(all_numbers),
                    "suspect_numbers": max(0, len(all_numbers) - len(explicit_units))
                },
                "csv_structure": {
                    "headers": header,
                    "analyzed_variables": analyzed_variables,
                    "preview_rows_count": len(preview_rows)
                }
            }
            
        except Exception:
            explicit_units = [f"{v[0]} {v[1]}" for v in self.numeric_unit_pattern.findall(full_body_content)]
            all_numbers = self.generic_numeric_regex.findall(full_body_content)
            semantic_metrics = self._compute_semantic_metrics(full_body_content)
            anchor_correlations = self._compute_anchor_correlations(full_body_content)
            
            return {
                "source": "Unstructured Flat File",
                "title": f"Raw Text Audit: {filename}",
                "filename": filename,
                "variables_found": explicit_units,
                "semantic_metrics": semantic_metrics,
                "anchor_correlations": anchor_correlations,
                "stats": {
                    "total_numbers": len(all_numbers),
                    "suspect_numbers": max(0, len(all_numbers) - len(explicit_units))
                },
                "raw_text_preview": full_body_content[:400] + "..."
            }

    def analyze_file(self, rel_filepath):
        """Smista l'analisi preservando e ricostruendo la sottocartella del sub-corpus."""
        # Se viene passato un percorso relativo, os.path.join gestisce la concatenazione
        path = os.path.join(self.source_dir, rel_filepath)
        filename = os.path.basename(rel_filepath)
        
        if filename.endswith(".html"):
            data = self.process_html(path, filename)
        elif filename.endswith((".csv", ".txt")):
            data = self.process_csv_txt(path, filename)
        else:
            return None
            
        # Determina la sottocartella del sub-corpus per salvare il JSON in modo speculare
        subfolder = os.path.dirname(rel_filepath)
        target_dir = os.path.join(self.output_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        
        output_filename = filename.rsplit('.', 1)[0] + ".json"
        with open(os.path.join(target_dir, output_filename), "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, indent=2, ensure_ascii=False)
            
        return data

    def generate_evolution_report(self):
        """
        Aggrega i dati estratti dai 4 campioni fisici per generare la matrice tabellare
        dell'evoluzione coerente della TA ed esportare il grafico delle Fasi per il web.
        """
        LOG.info("=== Generazione Tabella Evolutiva e Grafici Fasi TA ===")
        
        # Mappatura formale tra le sottocartelle del sub-corpus e le 4 Fasi Teoretiche
        fasi_mapping = {
            "campione_1_lightning_neutron": "Fase 1 - Preparazione elettrica",
            "campione_2_tgf": "Fase 2 - Conversione fotonica (TGF)",
            "campione_3_neutron_burst_thunderstorm": "Fase 3 - Rilascio nucleare (Neutron burst)",
            "campione_4_atmospheric_plasma_discharge": "Fase 4 - Dissipazione plasmica"
        }
        semantic_phase_mapping = {
            "electric": "Fase 1 - Preparazione elettrica",
            "lightning": "Fase 2 - Conversione fotonica (TGF)",
            "nuclear": "Fase 3 - Rilascio nucleare (Neutron burst)",
            "temporal": "Fase 4 - Dissipazione plasmica"
        }
        
        fasi_data = {fase: {c: 0.0 for c in ANCHOR_CONCEPTS} for fase in fasi_mapping.values()}
        fasi_counts = {fase: 0 for fase in fasi_mapping.values()}
        
        if not os.path.exists(self.output_dir):
            return
            
        for root, _, files in os.walk(self.output_dir):
            folder_name = os.path.basename(root)
            folder_phase = fasi_mapping.get(folder_name)
            for file in files:
                if file.endswith(".json") and file != "summary.json":
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            data = json.load(f)
                        semantic_metrics = data.get("semantic_metrics", {})
                        dominant_semantic = max(
                            semantic_metrics,
                            key=lambda key: semantic_metrics.get(key, 0)
                        ) if semantic_metrics else None
                        if dominant_semantic and semantic_metrics.get(dominant_semantic, 0) <= 0:
                            dominant_semantic = None
                        fase_attuale = folder_phase or semantic_phase_mapping.get(dominant_semantic)

                        if not fase_attuale:
                            continue

                        # --- FIX DISALLINEAMENTO CHIAVI REPORT EVOLUTIVO ---
                        correlations = data.get("anchor_correlations", {})
                        for concept, score in correlations.items():
                            # Se la fase attuale del report non ha ancora questa chiave, la inizializza
                            if concept not in fasi_data[fase_attuale]:
                                fasi_data[fase_attuale][concept] = 0.0
                            fasi_data[fase_attuale][concept] += score
                        fasi_counts[fase_attuale] += 1
                    except Exception as e:
                        LOG.error(f"Errore nella lettura del file {file} per report evolutivo: {e}")

        report_rows = []
        for fase, concetti in fasi_data.items():
            count = fasi_counts[fase]
            row = {"Fase Evolutiva": fase}
            for concetto in ANCHOR_CONCEPTS:
                sommatoria = concetti.get(concetto, 0.0)
                row[concetto] = round(sommatoria / count, 2) if count > 0 else 0.0
            report_rows.append(row)
            
        df_evolution = pd.DataFrame(report_rows).set_index("Fase Evolutiva")
        # ... da qui prosegui con il salvataggio in CSV e il grafico plt.plot() ciclando su tutte_le_chiavi_reali
        evolution_csv = os.path.join(self.reports_dir, "ta_evolution_phases.csv")
        df_evolution.to_csv(evolution_csv)
        LOG.info(f"Tabella dell'evoluzione coerente salvata in: {evolution_csv}")
        
        # === Generazione Grafico Evolutivo Lineare (PNG per Interfaccia Web Dashboard) ===
        try:
            plt.figure(figsize=(10, 6))
            for concetto in ANCHOR_CONCEPTS:
                plt.plot(df_evolution.index, df_evolution[concetto], marker='o', linewidth=2.5, label=concetto.split(": ")[1])
                
            plt.title("Evoluzione Coerente dei Concetti Ancoranti nei Dataset (TA)", fontsize=14, weight='bold', pad=15)
            plt.ylabel("Grado di Correlazione Semantico-Vettoriale (0-1)", fontsize=12)
            plt.xlabel("Fasi di Transizione del Dataset Atmosferico", fontsize=12)
            plt.xticks(rotation=15)
            plt.ylim(-0.05, 1.05)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(loc="upper left", frameon=True, shadow=True)
            plt.tight_layout()
            
            evolution_png = os.path.join(self.reports_dir, "evolution_phases.png")
            plt.savefig(evolution_png, dpi=300)
            plt.close()
            LOG.info(f"Grafico evolutivo PNG salvato in: {evolution_png}")
        except Exception as graph_err:
            LOG.error(f"Impossibile generare il grafico delle fasi: {graph_err}")
