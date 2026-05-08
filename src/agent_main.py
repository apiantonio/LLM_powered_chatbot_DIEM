import sys
import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

# Importazioni dell'ecosistema DIEM RAG
from config.settings import load_settings
from ingestion.indexer import KnowledgeBaseIndexer
from agent.executor import build_conversational_agent

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_Orchestrator")

def run_pipeline():
    """
    Esegue la pipeline completa:
    1. Caricamento settings.
    2. Indicizzazione Knowledge Base (HTML & PDF).
    3. Avvio del loop conversazionale dell'Agente.
    """
    print("="*50)
    print("Inizializzazione DIEM RAG System Architecture")
    print("="*50)
    
    # --- STEP 1: Lettura configurazioni ---
    logger.info("Caricamento configurazioni di sistema...")
    settings = load_settings()

    # --- STEP 2: Indicizzazione (Ingestion) ---
    print("\n[1/3] Avvio processo di indicizzazione sui Vectorstore...")
    try:
        indexer = KnowledgeBaseIndexer(settings)
        
        logger.info("Elaborazione dati HTML in corso...")
        html_stats = indexer.index_html_directory()
        print(f"  -> Statistiche Indexing HTML: {html_stats}")
        
        logger.info("Elaborazione e partizionamento dati PDF in corso...")
        pdf_stats = indexer.index_pdf_list()
        print(f"  -> Statistiche Indexing PDF: {pdf_stats}")
        
    except Exception as e:
        logger.error(f"Errore critico durante l'indicizzazione: {e}")
        sys.exit(1)

    # --- STEP 3: Costruzione Agent RAG ---
    print("\n[2/3] Costruzione Agente Conversazionale...")
    try:
        agent = build_conversational_agent(settings)
    except Exception as e:
        logger.error(f"Fallimento durante l'inizializzazione dell'agente: {e}")
        sys.exit(1)

    # --- STEP 4: Chat Loop Interattivo ---
    print("\n[3/3] Sistema Online. Pronti all'interazione.")
    print("\n" + "="*50)
    print("Assistente DIEM - Loop Interattivo Avviato")
    print("Digita 'exit' o 'quit' per terminare la sessione.")
    print("="*50 + "\n")

    session_id = "cli_production_session_001"
    run_config: Dict[str, Any] = {
        "configurable": {
            "session_id": session_id
        }
    }

    while True:
        try:
            user_input = input("\n👤 Tu: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nAssistente DIEM: Arrivederci e grazie per aver usato il sistema!")
                break
                
            if not user_input:
                continue

            print("🤖 Assistente DIEM sta elaborando la richiesta in base al contesto e alle fonti...")
            
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=run_config
            )
            
            final_message = response["messages"][-1].content
            print(f"\nAssistente DIEM:\n{final_message}")

        except KeyboardInterrupt:
            print("\n\nAssistente DIEM: Sessione interrotta forzatamente. Termino processo.")
            break
        except Exception as e:
            logger.error(f"Impossibile processare la richiesta dell'utente a causa di un errore nel grafo: {e}")

if __name__ == "__main__":
    run_pipeline()