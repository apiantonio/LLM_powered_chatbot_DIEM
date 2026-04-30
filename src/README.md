```
├── src/                            # Codice sorgente core dell'applicazione
   ├── __init__.py
   ├── ingestion/                   # [Modulo 1] Pipeline di caricamento dati offline
   │   ├── __init__.py
   │   ├── scrapers.py              # Logica per RecursiveUrlLoader e pulizia HTML personalizzata
   │   ├── document_loaders.py      # Gestione PyPDFLoader e parser vari
   │   └── indexer.py               # Logica di chunking (es. HTMLSectionSplitter) ed embedding
   ├── retrieval/                   # [Modulo 2] Motore di ricerca avanzato
   │   ├── __init__.py
   │   ├── reranker.py              # Implementazione del Cross-Encoder per il re-ranking post-retrieval
   │   └── query_optimizer.py       # Funzioni di Query Rewriting e Multi-Query pre-retrieval
   ├── agent/                       # [Modulo 3] Orchestrazione dell'Agente LLM
   │   ├── __init__.py
   │   ├── tools.py                 # Definizione dei @tool (Ricerca Vettoriale, Orari EasyCourse)
   │   ├── prompts.py               # System prompt rigorosi, direttive di Scope Awareness
   │   └── executor.py              # Inizializzazione dell'Agente (es. tramite create_agent) e memoria
   ├── evaluation/                  # [Modulo 4] Pipeline di testing quantitativo
   │   ├── __init__.py
   │   └── ragas_runner.py          # Script di automazione per il calcolo delle metriche RAGAS
   └── config/                      # Configurazione centralizzata
       ├── __init__.py
       └── settings.py              # Gestione delle variabili d'ambiente e parametri globali (chunk_size, LLM model)
```