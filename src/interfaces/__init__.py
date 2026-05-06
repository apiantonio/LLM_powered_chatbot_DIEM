"""
Interfacce astratte (Ports) del sistema RAG.

Design Pattern: Strategy (GoF) + Dependency Inversion (SOLID-D).
Ogni componente dipende da queste astrazioni, mai dalle implementazioni concrete.
Questo permette di:
- Cambiare LLM provider senza toccare la logica dell'agente.
- Sostituire il Vector Store senza modificare il retrieval engine.
- Testare ogni modulo in isolamento con mock/stub.

KPI Impact: Ingegneria del Software (modularità, testabilità, estendibilità).
"""