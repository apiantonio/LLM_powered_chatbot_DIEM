import sys
from typing import Dict, Any

# LangChain core imports
from langchain_core.messages import HumanMessage

# Importazione dell'orchestratore costruito nel Task 3.2
from src.agent.executor import build_conversational_agent

def chat_loop():
    """
    Inizializza l'agente RAG e avvia un loop interattivo REPL su riga di comando.
    Gestisce la sessione utente per dimostrare la persistenza in-memory del contesto.
    """
    print("Inizializzazione del DIEM Agentic RAG in corso...")
    
    try:
        # Inizializziamo l'agente (il factory pattern ci restituisce il Runnable configurato)
        agent = build_conversational_agent()
    except Exception as e:
        print(f"[ERROR] Fallimento critico durante l'inizializzazione dell'agente: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("Assistente DIEM Online.")
    print("Digita 'exit' o 'quit' per terminare la sessione.")
    print("="*50 + "\n")

    # Hardcodiamo un session_id per questa iterazione della CLI.
    # In produzione, questo ID deriverebbe dal token JWT o dalla sessione web dell'utente.
    session_id = "cli_session_001"
    
    # Configurazione runtime richiesta da RunnableWithMessageHistory
    run_config: Dict[str, Any] = {
        "configurable": {
            "session_id": session_id
        }
    }

    while True:
        try:
            user_input = input("\n👤 Tu: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Assistente DIEM: Arrivederci!")
                break
                
            if not user_input:
                continue

            print("Assistente DIEM sta elaborando...")
            
            # Invocazione del grafo. Poiché usiamo create_agent, l'input_messages_key
            # configurata in RunnableWithMessageHistory mappa questa lista di messaggi 
            # direttamente allo StateGraph interno.
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=run_config
            )
            
            # Estraiamo l'output finale. create_agent restituisce il dizionario di stato 
            # completo. L'ultimo messaggio nella lista "messages" è la risposta dell'AI.
            final_message = response["messages"][-1].content
            
            print(f"\nAssistente DIEM: {final_message}")

        except KeyboardInterrupt:
            print("\nAssistente DIEM: Sessione interrotta forzatamente. Arrivederci!")
            break
        except Exception as e:
            # Fallback robusto in caso di errore di inference (es. API HuggingFace down)
            print(f"\n[SYSTEM ERROR] Impossibile processare la richiesta: {e}")

if __name__ == "__main__":
    chat_loop()