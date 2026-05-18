from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import load_settings

settings = load_settings()

embedding_model = HuggingFaceEmbeddings(
    model_name=settings.embedding.model_name,
    encode_kwargs={"normalize_embeddings": settings.embedding.normalize_embeddings}
)

vectorstore = Chroma(
    collection_name="dipartimento",
    embedding_function=embedding_model,
    persist_directory=settings.vectorstore.persist_directory
)

risultati = vectorstore.get(where={"sotto_area": "internazionale"})

metadati = risultati.get("metadatas", [])
sorgenti_univoche = set()

for m in metadati:
    sorgente = m.get("source_url")
    if not sorgente:
        sorgente = m.get("source_file")

    if sorgente:
        sorgenti_univoche.add(sorgente)

for sorgente in sorgenti_univoche:
    print(sorgente)