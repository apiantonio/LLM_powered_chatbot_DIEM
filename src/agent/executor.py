import os
from typing import List

# Nuove importazioni basate sulle API aggiornate di LangChain
from langchain.agents import create_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool

# LLM stack (HuggingFace)
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Importazioni moduli interni
from src.agent.prompts import get_agent_system_prompt
from src.agent.memory import get_session_history
from src.agent.tools import get_diem_tools 

def build_conversational_agent(
    repo_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    temperature: float = 0.1,
    max_new_tokens: int = 512
) -> RunnableWithMessageHistory:
    """
    Assembla e restituisce l'agente conversazionale RAG utilizzando il costruttore a grafo create_agent.
    """
    
    # 1. Inizializzazione del LLM
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        task="text-generation",
        do_sample=True if temperature > 0 else False
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)
    
    # 2. Recupero dei Tool e del SystemMessage
    tools: List[BaseTool] = get_diem_tools()
    system_prompt = get_agent_system_prompt()
    
    # 3. Creazione del Grafo dell'Agente tramite API nativa LangChain
    # Rimpiazza completamente AgentExecutor gestendo in autonomia il loop ReAct
    agent_graph = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=system_prompt,
        debug=True # Abilita il log verboso delle transizioni di stato (sostituisce verbose=True)
    )
    
    # 4. Integrazione della Memoria Conversazionale
    # create_agent si aspetta e aggiorna una chiave di stato chiamata "messages".
    # Utilizziamo RunnableWithMessageHistory per appendere automaticamente la history passata 
    # e l'input utente a questa chiave.
    conversational_agent = RunnableWithMessageHistory(
        runnable=agent_graph,
        get_session_history=get_session_history,
        input_messages_key="messages",
        history_messages_key="messages", # Mappiamo entrambe alla stessa chiave attesa dal Grafo
    )
    
    return conversational_agent