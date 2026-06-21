import os
import sys
import json
import time
import threading
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Previene conflitti di backend sui thread di Flask

from flask import Flask, Response, render_template, jsonify, request, send_from_directory
from src.ingestion.downloader import AdaptiveDownloader
from src.processing.extractor import InformationExtractor
from src.utils.metrics import SemanticsEngine
from src.learning.clusterer import IntelligenceClusterer

app = Flask(__name__, template_folder='templates', static_folder='static')

EXTRACTED_DIR = "data/extracted/"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOG_FILE = "pipeline.log"

# Configurazione dizionario Campioni speculare a run_shell.py
CAMPIONI = {
    "campione_1_lightning_neutron": {"target": "lightning neutron", "specific": "neutron"},
    "campione_2_tgf": {"target": '"Terrestrial Gamma-ray Flash"', "specific": "tgf"},
    "campione_3_neutron_burst_thunderstorm": {"target": "thunderstorm neutron burst", "specific": "neutron burst"},
    "campione_4_atmospheric_plasma_discharge": {"target": "atmospheric plasma discharge", "specific": "plasma discharge"}
}

TOTAL_EXPECTED_SOURCES = (AdaptiveDownloader.TOTALE if hasattr(AdaptiveDownloader, 'TOTALE') else 40)

PIPELINE_STATUS = {
    "running": False,
    "current_stage": "In attesa...",
    "progress": 0,
    "completed_tasks": []
}

class RealTimeLogInterceptor(object):
    def __init__(self, current_stdout):
        self.terminal = current_stdout

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        clean_msg = message.replace('\r', '\n')
        if clean_msg.strip():
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(clean_msg)

    def flush(self):
        self.terminal.flush()

def get_stats():
    stats = {
        "total_sources": 0,
        "total_extracted": 0,
        "heatmap_exists": False,
        "evolution_exists": False,
        "semantic_gap_exists": False
    }
    
    # Conteggio ricorsivo sui 4 campioni fisici
    base_sources = "data/sources/"
    if os.path.exists(base_sources):
        for root, _, files in os.walk(base_sources):
            stats["total_sources"] += len([f for f in files if not f.startswith('.')])
            
    if os.path.exists(EXTRACTED_DIR):
        for root, _, files in os.walk(EXTRACTED_DIR):
            stats["total_extracted"] += len([f for f in files if f.endswith(".json") and f != "summary.json"])
            
    stats["heatmap_exists"] = os.path.exists(os.path.join(REPORTS_DIR, "correlation_heatmap.png"))
    stats["evolution_exists"] = os.path.exists(os.path.join(REPORTS_DIR, "evolution_phases.png"))
    stats["semantic_gap_exists"] = os.path.exists(os.path.join(REPORTS_DIR, "semantic_gap_score.png"))
    return stats

def run_pipeline_worker(mode):
    global PIPELINE_STATUS
    PIPELINE_STATUS["running"] = True
    PIPELINE_STATUS["completed_tasks"] = []
    
    try:
        extractor = InformationExtractor()
        
        # FASE 1: INGESTION ISOLATA NEI 4 CORPORA
        if mode == 'all' or mode == 'ingest':
            PIPELINE_STATUS["current_stage"] = "Fase 1: Ingestion Layer (Download Sub-Corpora)"
            PIPELINE_STATUS["progress"] = 0
            
            print("[Reset] Pulizia iniziale delle directory di lavoro...")
            # Rimozione vecchi asset grafici per evitare falsi positivi
            for img_name in [
                "correlation_heatmap.png",
                "evolution_phases.png",
                "semantic_gap_score.png",
                "ta_evolution_phases.csv",
                "semantic_gap_report.csv",
                "semantic_gap_report.json"
            ]:
                p = os.path.join(REPORTS_DIR, img_name)
                if os.path.exists(p):
                    os.remove(p)

            downloader = AdaptiveDownloader()
            downloader.clear_previous_sessions()
            
            for folder, queries in CAMPIONI.items():
                print(f"[Ingestion Web] Download controllato per sub-corpus: {folder}")
                downloader.set_subfolder(folder)
                downloader.fetch_arxiv_and_ar5iv(queries["target"])
                downloader.fetch_zenodo(queries["target"])
                downloader.fetch_nasa_ads(queries["specific"])
                downloader.fetch_noaa(queries["target"])
                downloader.fetch_openaire(queries["target"])
                downloader.fetch_figshare(queries["target"])
                downloader.fetch_kaggle(queries["target"])
                PIPELINE_STATUS["completed_tasks"].append(f"Sincronizzato sub-corpus: {folder}")
            
            PIPELINE_STATUS["completed_tasks"].append("FASE 1: Ingestion Layer Completata")
            time.sleep(1)

        # FASE 2: PROCESSING CON PRESERVAZIONE DEL PATH RELATIVO
        if mode == 'all' or mode == 'process':
            PIPELINE_STATUS["current_stage"] = "Fase 2: Processing Layer (Mappatura Concetti Ancoranti)"
            PIPELINE_STATUS["progress"] = 0
            
            base_sources = "data/sources/"
            all_valid_files = []
            
            if os.path.exists(base_sources):
                for root, _, files in os.walk(base_sources):
                    for file in files:
                        if file.endswith((".txt", ".html", ".csv")):
                            rel_path = os.path.relpath(os.path.join(root, file), base_sources)
                            all_valid_files.append(rel_path)
            
            total_files = len(all_valid_files)
            if total_files == 0:
                print("[ERRORE] Nessun file sorgente rilevato nelle sottocartelle dei campioni.")
                print("[DONE]")
                PIPELINE_STATUS["running"] = False
                return
                
            print(f"[Pipeline Web] Avvio estrazione su {total_files} file distribuiti...")
            for idx, rel_filepath in enumerate(all_valid_files, start=1):
                extractor.analyze_file(rel_filepath)
                PIPELINE_STATUS["progress"] = int((idx / total_files) * 100)
            
            print("[Pipeline Web] Compilazione report evolutivo (Matrice delle 4 Fasi TA)...")
            extractor.generate_evolution_report()
            
            PIPELINE_STATUS["completed_tasks"].append("FASE 2: Estrazione, Anchor Mapping e Grafico Fasi completati")
            time.sleep(1)

        # FASE 3: LEARNING & ANALYTICS GLOBAL
        if mode == 'all' or mode == 'learn':
            PIPELINE_STATUS["current_stage"] = "Fase 3: Calcolo Matrice Pearson e Clustering AI"
            PIPELINE_STATUS["progress"] = 30
            
            # Se siamo andati diretti alla fase 3, rigeneriamo comunque l'evolution report sui dati estratti esistenti
            extractor.generate_evolution_report()
            
            engine = SemanticsEngine()
            engine.calculate_correlations()
            PIPELINE_STATUS["progress"] = 70
            
            clusterer = IntelligenceClusterer()
            clusterer.perform_clustering(n_clusters=2)
            
            PIPELINE_STATUS["progress"] = 100
            PIPELINE_STATUS["completed_tasks"].append("FASE 3: Analisi completata. Asset grafici pronti per la Dashboard")

        print("[DONE]")
        PIPELINE_STATUS["current_stage"] = "Tutti i processi sono terminati con successo!"
    except Exception as e:
        print(f"[ERRORE CRITICO PIPELINE]: {str(e)}")
        print("[DONE]")
        PIPELINE_STATUS["current_stage"] = f"Blocco per Errore: {str(e)}"
    finally:
        PIPELINE_STATUS["running"] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reports/<filename>')
def serve_reports(filename):
    return send_from_directory(REPORTS_DIR, filename)

@app.route('/api/run/pipeline', methods=['POST'])
def start_pipeline_thread():
    mode = request.args.get('mode', 'all')
    if os.path.exists(LOG_FILE):
        try: os.remove(LOG_FILE)
        except: pass
    
    global PIPELINE_STATUS
    PIPELINE_STATUS["running"] = True
    PIPELINE_STATUS["progress"] = 0
    PIPELINE_STATUS["current_stage"] = "Inizializzazione Thread asincrono..."
    PIPELINE_STATUS["completed_tasks"] = []

    t = threading.Thread(target=run_pipeline_worker, args=(mode,))
    t.start()
    return jsonify({"status": "Thread attivato correttamente"})

@app.route('/api/status')
def get_pipeline_status():
    global PIPELINE_STATUS
    return jsonify({
        "running": PIPELINE_STATUS["running"],
        "current_stage": PIPELINE_STATUS["current_stage"],
        "progress": PIPELINE_STATUS["progress"],
        "completed_tasks": PIPELINE_STATUS["completed_tasks"],
        "total_expected": TOTAL_EXPECTED_SOURCES,
        "stats": get_stats()
    })

@app.route('/api/logs/live')
def read_live_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype='text/plain')
    return Response("", mimetype='text/plain')

@app.route('/data/extracted/<path:filename>')
def serve_extracted_files(filename):
    return send_from_directory(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/extracted')), filename)

if __name__ == '__main__':
    sys.stdout = RealTimeLogInterceptor(sys.stdout)
    print("[Interface Layer] Avvio della Dashboard Web Intelligente sulla porta 5000...")
    app.run(debug=True, port=5000, use_reloader=False)
