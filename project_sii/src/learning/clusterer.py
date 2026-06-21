"""
src/learning/clusterer.py
Modulo Learning Layer: Vettorizzazione TF-IDF e Machine Learning K-Means
per la categorizzazione non supervisionata dei transitori fisici.
"""

import os
import json
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.memory.schema import DATA_EXTRACTED, REPORTS_DIR

LOG = logging.getLogger("SII_CLUSTERER")

class IntelligenceClusterer:
    def __init__(self, extracted_dir=DATA_EXTRACTED, reports_dir=REPORTS_DIR):
        self.extracted_dir = extracted_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def perform_clustering(self, n_clusters=2):
        """
        Esegue il clustering non supervisionato K-Means sul corpus estratto ricorsivamente,
        isolando la dicotomia semantica latente nella letteratura scientifica.
        """
        LOG.info(f"=== Avvio Modulo Learning: Clustering K-Means (n_clusters={n_clusters}) ===")
        
        corpus = []
        valid_files = []
        
        if not os.path.exists(self.extracted_dir):
            LOG.error(f"Directory dei dati estratti '{self.extracted_dir}' non trovata. Clustering interrotto.")
            return

        # Scansione ricorsiva per mappare i 4 corpora indipendenti suddivisi in sottocartelle
        for root, _, files in os.walk(self.extracted_dir):
            for filename in sorted(files):
                if not filename.endswith(".json") or filename == "summary.json":
                    continue
                    
                path = os.path.join(root, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    text_content = ""
                    
                    # Caso 1: Sezione HTML reali (arXiv / ar5iv)
                    if "html_sections" in data and isinstance(data["html_sections"], list):
                        text_content += " ".join([sec.get("text", "") or sec.get("caption", "") for sec in data["html_sections"]])
                    
                    # Caso 2: Anteprima testuale o fallback
                    if "raw_text_preview" in data and isinstance(data["raw_text_preview"], str):
                        text_content += " " + data["raw_text_preview"]
                        
                    # Caso 3: Chiavi legacy per retrocompatibilità
                    if "raw_text_clean" in data and isinstance(data["raw_text_clean"], str):
                        text_content += " " + data["raw_text_clean"]

                    # Caso 4: Integrazione con i campi estratti dal SemanticsEngine (paragrafi e didascalie)
                    for key in ["key_paragraphs", "fig_captions"]:
                        if key in data and isinstance(data[key], list):
                            text_content += " " + " ".join(data[key])
                            
                    # Caso 5: Tabelle nidificate
                    if "tables" in data and isinstance(data["tables"], list):
                        for t in data["tables"]:
                            if isinstance(t, dict) and t.get("caption"):
                                text_content += " " + t["caption"]
                            elif isinstance(t, str):
                                text_content += " " + t

                    text_content = text_content.strip()
                    
                    if text_content:
                        corpus.append(text_content)
                        # Conserviamo il path relativo per tracciare a quale campione appartiene il file
                        valid_files.append(os.path.relpath(path, self.extracted_dir))
                    else:
                        # Fallback per CSV puri: usiamo gli header strutturali come token testuali
                        if "csv_structure" in data and isinstance(data["csv_structure"], dict):
                            csv_text = " ".join(data["csv_structure"].get("headers", []))
                            if csv_text.strip():
                                corpus.append(csv_text)
                                valid_files.append(os.path.relpath(path, self.extracted_dir))
                                
                except Exception as e:
                    LOG.error(f"Errore durante la lettura del file {filename} per il clustering: {e}")

        if not corpus:
            LOG.error("[Learning - ERROR] Corpus vuoto. Impossibile addestrare il K-Means.")
            return

        # Controllo di sicurezza sulle dimensioni minime del campione rispetto ai centroidi richiesti
        if len(corpus) < n_clusters:
            LOG.warning(f"Numero di documenti validi ({len(corpus)}) inferiore al numero di cluster richiesti ({n_clusters}). Riduzione automatica di n_clusters.")
            n_clusters = len(corpus)

        # === 1. Estrazione delle Feature tramite Vettorizzazione TF-IDF ===
        # max_df=0.95 rimuove le stopword specifiche del dominio (es. parole troppo comuni in tutti i paper)
        # min_df=2 assicura che il termine compaia in almeno due documenti se il corpus lo consente
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
        X = vectorizer.fit_transform(corpus)
        
        # === 2. Addestramento del Modello Non Supervisionato K-Means ===
        # random_state fissato a 42 per garantire la riproducibilità esatta dei risultati ad ogni esecuzione
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_

        # === 3. Organizzazione dei Risultati in un DataFrame ed Esportazione ===
        results_df = pd.DataFrame({
            "file": valid_files,
            "cluster_assigned": labels
        }).set_index("file")
        
        clustering_csv = os.path.join(self.reports_dir, "clustering_results.csv")
        results_df.to_csv(clustering_csv)
        LOG.info(f"Risultati del clustering salvati in: {clustering_csv}")

        # === 4. Analisi dei Centroidi: Estrazione delle Parole Chiave Dominanti per Cluster ===
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        
        LOG.info("--- Top keywords per ciascun Cluster identificato ---")
        cluster_summary = {}
        for i in range(n_clusters):
            # Isola i primi 10 termini con il peso TF-IDF più elevato nel centroide
            top_words = [terms[ind] for ind in order_centroids[i, :10]]
            LOG.info(f"Cluster {i}: {', '.join(top_words)}")
            cluster_summary[f"cluster_{i}_keywords"] = top_words

        # Esporta un piccolo dizionario di riepilogo in JSON per l'ispezione automatica o web
        summary_path = os.path.join(self.reports_dir, "clustering_summary.json")
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(cluster_summary, sf, indent=2, ensure_ascii=False)
        LOG.info(f"Riepilogo delle keyword dei centroidi salvato in: {summary_path}")