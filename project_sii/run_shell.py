import sys
import os
import matplotlib
matplotlib.use('Agg')  # Forzza il backend headless prima di caricare qualsiasi modulo interno
# Assicura la corretta risoluzione dei moduli interni
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.downloader import AdaptiveDownloader
from src.processing.extractor import InformationExtractor
from src.processing.verifier import PipelineVerifier
from src.utils.metrics import SemanticsEngine
from src.learning.clusterer import IntelligenceClusterer

def show_menu():
    print("\n" + "="*60)
    print("SISTEMA INTELLIGENTE DI INFORMATION EXTRACTION & TEXT MINING (SII)")
    print("Autore: Douglas Ruffini - Mat: 482379")
    print("="*60)
    print("1. Esegui intera pipeline automaticamente")
    print("2. Scarica nuovi dati (Ingestion)")
    print("3. Estrai informazioni da fonti scaricate (Processing)")
    print("4. Rianalizza directory ed esegui Clustering (Learning)")
    print("0. Esci")
    print("="*60)

def downloading(downloader, target_query, specific_query, index, subfolder=None):
    """
    Funzione di download estesa per supportare il salvataggio mirato nei 4 campioni 
    indipendenti mediante il parametro opzionale subfolder.
    """
    if index == 1:
        downloader.clear_previous_sessions()    
    elif index == 2:
        # Configura la sottodirectory specifica nel downloader se passata
        if subfolder and hasattr(downloader, 'set_subfolder'):
            downloader.set_subfolder(subfolder)
            
        # Download mirato per sorgente
        downloader.fetch_arxiv_and_ar5iv(target_query)
        downloader.fetch_zenodo(target_query)
        downloader.fetch_nasa_ads(specific_query)
        downloader.fetch_noaa(target_query)
        downloader.fetch_openaire(target_query)
        downloader.fetch_figshare(target_query)
        downloader.fetch_kaggle(target_query)
    elif index == 3:
        if subfolder and hasattr(downloader, 'set_subfolder'):
            downloader.set_subfolder(subfolder)
        downloader.generate_readme(target_query, specific_query)
   
    return True

def main():
    downloader = AdaptiveDownloader()
    extractor = InformationExtractor()
    verifier = PipelineVerifier()
    engine = SemanticsEngine()
    clusterer = IntelligenceClusterer()
    
    # Definizione rigorosa dei 4 corpora tematici indipendenti (Specifiche Punto 2.3)
    CAMPIONI = {
        "campione_1_lightning_neutron": {
            "target": "lightning neutron",
            "specific": "lightning discharge neutron"
        },
        "campione_2_tgf": {
            "target": '"Terrestrial Gamma-ray Flash" lightning',
            "specific": "tgf lightning transient"
        },
        "campione_3_neutron_burst_thunderstorm": {
            "target": '"neutron burst" thunderstorm',
            "specific": "thunderstorm nuclear burst"
        },
        "campione_4_atmospheric_plasma_discharge": {
            "target": '"atmospheric plasma" discharge neutron gamma',
            "specific": "plasma high energy discharge"
        }
    }

    while True:
        show_menu()
        scelta = input("Seleziona un'opzione [0-4]: ").strip()
        
        if scelta == "1":
            print("\n[Pipeline] Inizializzazione Ambiente...")
            downloading(downloader, None, None, 1)
            
            # Ciclo iterativo sui 4 corpora indipendenti richiesti dal Processing Layer
            for folder, queries in CAMPIONI.items():
                print(f"\n[Pipeline] Ingestion Layer -> Elaborazione {folder}...")
                # Download isolato nella propria cartella di riferimento
                downloading(downloader, queries["target"], queries["specific"], 2, subfolder=folder)
                downloading(downloader, queries["target"], queries["specific"], 3, subfolder=folder)
                
                # Processing Layer (Information Extraction) mirato sulla sottocartella specifica
                source_path = os.path.join("data", "sources", folder)
                if os.path.exists(source_path):
                    for file in os.listdir(source_path):
                        if file.endswith((".txt", ".html", ".csv")):
                            # Passiamo il path relativo corretto includendo il sub-corpus
                            extractor.analyze_file(os.path.join(folder, file))
            
            # === ANALYTICS & LEARNING LAYER PIPELINE ===
            print("\n[Pipeline] Generazione report evolutivo delle Fasi della Teoria dell'Avvicinamento...")
            extractor.generate_evolution_report()  # Genera 'ta_evolution_phases.csv' e 'evolution_phases.png'
            
            print("\n[Pipeline] Avvio del Verifier e auditing dei dati numerici...")
            verifier.generate_summary()             # Genera 'summary.json' e 'summary.html'
            
            print("\n[Pipeline] Calcolo delle correlazioni di Pearson...")
            engine.calculate_correlations()         # Genera 'semantic_correlation.csv' e 'correlation_heatmap.png'
            
            print("\n[Pipeline] Avvio del modulo di Learning (Clustering non supervisionato K-Means)...")
            clusterer.perform_clustering(n_clusters=2) # Genera 'clustering_results.csv' e 'clustering_summary.json'
            
            print("\n" + "="*60)
            print("PIPELINE STRUTTURATA E COMPLETATA CON SUCCESSO!")
            print("Tutti i report tabellari e grafici per il Web sono pronti in 'reports/'")
            print("="*60)

        elif scelta == "2":
            query = input("Inserisci query di ricerca manuale: ").strip()
            # Default sul comportamento flat standard se richiamato singolarmente
            downloading(downloader, query if query else 'lightning neutron', query if query else 'neutron', 2)
            
        elif scelta == "3":
            print("[Pipeline] Estrazione in corso da data/sources/...")
            # Gestione ricorsiva per preservare l'estrazione dai 4 campioni se presenti
            base_sources = "data/sources/"
            if os.path.exists(base_sources):
                for root, dirs, files in os.walk(base_sources):
                    for file in files:
                        if file.endswith((".txt", ".html", ".csv")):
                            # Ricava il percorso relativo rispetto a data/sources/
                            rel_path = os.path.relpath(os.path.join(root, file), base_sources)
                            extractor.analyze_file(rel_path)

            # === ALLINEAMENTO COMPLETO ANALYTICS ANCHE PER L'OPZIONE SOLO ANALISI ===
            print("\n[Pipeline] Generazione report evolutivo delle Fasi della Teoria dell'Avvicinamento...")
            extractor.generate_evolution_report()
            
            print("\n[Pipeline] Calcolo delle correlazioni di Pearson...")
            engine.calculate_correlations()
            
            print("\n[Pipeline] Esecuzione Clustering K-Means...")
            clusterer.perform_clustering(n_clusters=2)
            
            summary = verifier.generate_summary()
            print(f"\n[Processing] Analisi completata. File complessivi verificati: {summary.get('n_files', 0)}")
            print("Grafici e matrici aggiornati con successo.")
                     
        elif scelta == "4":
            print("[Pipeline] Elaborazione statistiche avanzate, grafici e clustering...")
            engine.calculate_correlations()
            clusterer.perform_clustering(n_clusters=2)
            
        elif scelta == "0":
            print("Chiusura dell'applicazione. Arrivederci Douglas.")
            break
        else:
            print("Opzione non valida. Riprova.")

if __name__ == "__main__":
    main()