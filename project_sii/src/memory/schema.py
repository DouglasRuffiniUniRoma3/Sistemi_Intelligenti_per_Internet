"""
src/memory/schema.py
Definizione del Thesaurus controllato T, delle categorie semantiche
e dei vincoli analitici del sistema SII.
"""

# Thesaurus Controllato T (Specifiche Punto 2.3, 3.1 e 3.2)
TERMS = {
    "electric": ["electric", "field", "charge", "voltage", "current"],
    "lightning": ["lightning", "storm", "thunder", "discharge"],
    "nuclear": ["neutron", "gamma", "flux", "tgf", "radiation"],
    "temporal": ["delay", "pulse", "relaxation", "decay", "lifetime", "duration", "time constant"]
}

# Cartelle di input/output predefinite rispetto alla root del progetto
DATA_EXTRACTED = "data/extracted"
REPORTS_DIR = "reports"

# Sottovettori semantici per l'algoritmo LLMatch-like dei Concetti Ancoranti (Teoria dell'Avvicinamento)
ANCHOR_CONCEPTS = {
    "Fase 1: Modello Oscillazione Smorzata": [
        "oscillation", "damped", "lorentz", "contraction", "velocity", "light", "acceleration", "collapse"
    ],
    "Fase 2: Modello RC Relativistico": [
        "rc", "time constant", "decay", "exponential", "heisenberg", "uncertainty", "scale", "lifetime"
    ],
    "Fase 3: Estensione Frattale e Numeri Primi": [
        "fractal", "prime", "goldbach", "recursive", "topology", "discrete", "space", "decomposition"
    ],
    "Fase 4: Sincronizzazione Transitori Energetici": [
        "transient", "co-occurrence", "neutron burst", "tgf", "gamma", "flux", "lightning", "discharge"
    ]
}