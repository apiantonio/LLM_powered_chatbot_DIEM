from typing import List
from langchain.agents import create_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool

# Importazioni moduli interni refattorizzati
from agent.prompts import get_agent_system_prompt
from agent.memory import get_session_history
from agent.tools import get_all_tools 
from agent.llm_providers import create_chat_model
from config.settings import AppSettings

def build_conversational_agent(
    settings: AppSettings
) -> RunnableWithMessageHistory:
    """
    Assembla e restituisce l'agente conversazionale RAG utilizzando il costruttore a grafo create_agent.
    I parametri operativi derivano dal single-source-of-truth AppSettings.
    """
    
    # 1. Inizializzazione del LLM centralizzata via Factory pattern
    chat_model = create_chat_model(settings.llm)
    
    # 2. Recupero dei Tool e del SystemMessage
    tools: List[BaseTool] = get_all_tools()
    system_prompt = get_agent_system_prompt()
    
    # 3. Creazione del Grafo dell'Agente
    agent_graph = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=system_prompt,
        debug=True 
    )
    
    # 4. Integrazione della Memoria Conversazionale
    conversational_agent = RunnableWithMessageHistory(
        runnable=agent_graph,
        get_session_history=get_session_history,
        input_messages_key="messages",
        history_messages_key="messages",
    )
    
    return conversational_agent