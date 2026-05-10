#!/usr/bin/env python3
"""
run_t6_4_ingestion.py — Script di Ingestion Definitiva (Task T6.4)

Esegue la re-indicizzazione COMPLETA della Knowledge Base DIEM nelle
4 collection Chroma multi-tematiche.

PREREQUISITI (già completati):
  ✅ T6.1 — Vector store Chroma svuotato
  ✅ T6.2 — Parent store svuotato
  ✅ T6.3 — index_registry.json resettato

QUESTO SCRIPT ESEGUE:
  1. Caricamento delle impostazioni centralizzate (config/settings.py)
  2. Inizializzazione del KnowledgeBaseIndexer multi-collection
  3. Indicizzazione HTML con routing automatico (DocumentRouter)
  4. Indicizzazione PDF con chunking differenziato
  5. Report finale con breakdown per collection e statistiche

Uso:
  cd src/
  python ingestion_main.py
  python ingestion_main.py --skip-crawl
  python ingestion_main.py --crawl
  python ingestion_main.py --log-level DEBUG
  python ingestion_main.py --verify-only
  python ingestion_main.py --log-dir my_logs
"""

import sys
import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

# ============================================================
# PATH SETUP
# ============================================================
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.settings import AppSettings, load_settings
from ingestion.indexer import KnowledgeBaseIndexer
from ingestion.router import CollectionTarget

logger = logging.getLogger(__name__)


# ============================================================
# VERIFICA POST-INGESTION
# ============================================================

def verify_collections(indexer: KnowledgeBaseIndexer, settings: AppSettings) -> Dict[str, Any]:
    """Verifica lo stato di ogni collection Chroma dopo l'ingestion."""
    verification: Dict[str, Any] = {"collections": {}, "total_chunks": 0, "ok": True}

    for target in CollectionTarget:
        # pylint: disable=protected-access
        collection = indexer._collections[target]
        try:
            data = collection.get()
            count = len(data["ids"]) if data and data.get("ids") else 0
        except Exception as e:
            logger.error(f"Errore verifica {target.value}: {e}")
            count = 0
            verification["ok"] = False

        verification["collections"][target.value] = count
        verification["total_chunks"] += count
        logger.info(f"  📊 {target.value}: {count} chunks")

    pc_collection_name = settings.vectorstore.parent_child_collection_name
    try:
        # pylint: disable=protected-access
        pc_data = indexer._pc_child_vectorstore.get()
        pc_count = len(pc_data["ids"]) if pc_data and pc_data.get("ids") else 0
    except Exception as e:
        logger.error(f"Errore verifica Parent-Child: {e}")
        pc_count = 0
        verification["ok"] = False

    verification["collections"][pc_collection_name] = pc_count
    verification["total_chunks"] += pc_count
    logger.info(f"  📊 {pc_collection_name} (Parent-Child childs): {pc_count} chunks")

    if verification["total_chunks"] == 0:
        verification["ok"] = False
        logger.warning("⚠️  ATTENZIONE: nessun chunk indicizzato!")

    return verification


def log_sample_documents(indexer: KnowledgeBaseIndexer, max_per_collection: int = 3) -> None:
    """Logga un campione di documenti per ogni collection per verifica visiva."""
    logger.info("\n" + "=" * 60)
    logger.info("🔎 CAMPIONE DOCUMENTI PER COLLECTION (verifica routing)")
    logger.info("=" * 60)

    for target in CollectionTarget:
        # pylint: disable=protected-access
        collection = indexer._collections[target]
        try:
            data = collection.get(include=["metadatas", "documents"])
            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])
            documents = data.get("documents", [])

            sample_size = min(max_per_collection, len(ids))
            logger.info(f"\n  📁 {target.value} ({len(ids)} chunks totali, mostro {sample_size}):")

            for i in range(sample_size):
                meta = metadatas[i] if i < len(metadatas) else {}
                content = documents[i][:120] if i < len(documents) and documents[i] else "(vuoto)"
                source = meta.get("source_url", "N/D")
                category = meta.get("doc_category", "N/D")
                doc_type = meta.get("doc_type", "N/D")
                logger.info(
                    f"    [{i+1}] type={doc_type} | cat={category}\n"
                    f"         source: {source}\n"
                    f"         content: {content}..."
                )
        except Exception as e:
            logger.warning(f"  ⚠️  Errore lettura campione {target.value}: {e}")


# ============================================================
# PIPELINE DI INGESTION T6.4
# ============================================================

def run_ingestion(settings: AppSettings, skip_crawl: bool = True) -> Dict[str, Any]:
    """
    Esegue l'ingestion completa T6.4:
      1. (Opzionale) Crawling dei siti DIEM
      2. Indicizzazione HTML con routing nelle 4 collection
      3. Indicizzazione PDF con chunking differenziato
      4. Verifica post-ingestion
    """
    start_time = time.time()
    report: Dict[str, Any] = {
        "task": "T6.4 — Re-indicizzazione completa multi-collection",
        "timestamp": datetime.now().isoformat(),
        "skip_crawl": skip_crawl,
        "crawl": None,
        "html_indexing": None,
        "pdf_indexing": None,
        "verification": None,
        "duration_seconds": 0,
        "success": False,
    }

    logger.info("=" * 70)
    logger.info("🚀 T6.4 — AVVIO RE-INDICIZZAZIONE COMPLETA MULTI-COLLECTION")
    logger.info("=" * 70)
    logger.info(f"   Timestamp: {report['timestamp']}")
    logger.info(f"   HTML dir: {settings.ingestion.html_raw_dir}")
    logger.info(f"   PDF links: {settings.ingestion.pdf_links_file}")
    logger.info(f"   Chroma dir: {settings.vectorstore.persist_directory}")
    logger.info(f"   Parent store: {settings.vectorstore.parent_store_directory}")
    logger.info(f"   Skip crawl: {skip_crawl}")
    logger.info("=" * 70)

    # --- STEP 0: Inizializzazione Indexer multi-collection ---
    logger.info("\n[STEP 0/4] Inizializzazione KnowledgeBaseIndexer multi-collection...")
    indexer = KnowledgeBaseIndexer(settings)

    logger.info("\n  Configurazioni chunking HTML per collection:")
    for target in CollectionTarget:
        chunk_size, chunk_overlap = settings.ingestion.get_collection_html_params(target.value)
        logger.info(f"    {target.value}: chunk_size={chunk_size}, overlap={chunk_overlap}")
    logger.info(
        f"  PDF Parent-Child: parent={settings.ingestion.pdf_parent_chunk_size}/"
        f"{settings.ingestion.pdf_parent_chunk_overlap}, "
        f"child={settings.ingestion.pdf_child_chunk_size}/"
        f"{settings.ingestion.pdf_child_chunk_overlap}"
    )
    logger.info(
        f"  PDF Diretto: {settings.ingestion.pdf_direct_chunk_size}/"
        f"{settings.ingestion.pdf_direct_chunk_overlap}"
    )

    # --- STEP 1: Crawling (opzionale) ---
    if not skip_crawl:
        logger.info("\n[STEP 1/4] Avvio crawling siti DIEM...")
        try:
            from ingestion.scrapers import UnisaCrawler

            html_rules = _load_html_cleaning_rules(settings)
            pdf_rules = _load_pdf_filter_rules(settings)

            crawler = UnisaCrawler(
                max_depth=settings.ingestion.max_depth,
                batch_size=settings.ingestion.batch_size,
                delay=settings.ingestion.crawl_delay_seconds,
                output_dir=settings.ingestion.html_raw_dir,
                html_rules=html_rules,
                pdf_rules=pdf_rules,
                crawler_config=settings.crawler,
                ingestion_config=settings.ingestion,
            )
            crawler.run()

            report["crawl"] = {
                "urls_visited": len(crawler.visited_urls),
                "html_saved": crawler.processed_count,
                "html_filtered": crawler.filtered_count,
                "pdf_links_found": len(crawler.found_pdf_links),
            }
            logger.info(f"  ✅ Crawling completato: {report['crawl']}")

        except ImportError as e:
            logger.warning(f"  ⚠️  Modulo crawling non disponibile: {e}")
            report["crawl"] = {"error": f"Import error: {e}"}
        except Exception as e:
            logger.error(f"  ❌ Errore durante il crawling: {e}", exc_info=True)
            report["crawl"] = {"error": str(e)}
    else:
        logger.info("\n[STEP 1/4] Crawling SALTATO (skip_crawl=True)")
        report["crawl"] = "skipped"

    # --- STEP 2: Indicizzazione HTML ---
    logger.info("\n[STEP 2/4] Indicizzazione HTML con routing multi-collection...")
    try:
        html_stats = indexer.index_html_directory()
        report["html_indexing"] = html_stats

        logger.info(f"  ✅ HTML indicizzazione completata:")
        logger.info(f"     Nuovi: {html_stats.get('indexed', 0)}")
        logger.info(f"     Aggiornati: {html_stats.get('updated', 0)}")
        logger.info(f"     Saltati (invariati): {html_stats.get('skipped', 0)}")
        logger.info(f"     Orfani rimossi: {html_stats.get('orphans_removed', 0)}")
        logger.info(f"     Errori: {html_stats.get('errors', 0)}")

        routing = html_stats.get("routing", {})
        if routing:
            logger.info("     Routing breakdown:")
            for coll_name, count in routing.items():
                logger.info(f"       📁 {coll_name}: {count} documenti")

    except Exception as e:
        logger.error(f"  ❌ Errore indicizzazione HTML: {e}", exc_info=True)
        report["html_indexing"] = {"error": str(e)}

    # --- STEP 3: Indicizzazione PDF ---
    logger.info("\n[STEP 3/4] Indicizzazione PDF con chunking differenziato...")
    try:
        pdf_stats = indexer.index_pdf_list()
        report["pdf_indexing"] = pdf_stats

        logger.info(f"  ✅ PDF indicizzazione completata:")
        logger.info(f"     Nuovi: {pdf_stats.get('indexed', 0)}")
        logger.info(f"     Aggiornati: {pdf_stats.get('updated', 0)}")
        logger.info(f"     Saltati (invariati): {pdf_stats.get('skipped', 0)}")
        logger.info(f"     Errori: {pdf_stats.get('errors', 0)}")
        logger.info(
            f"     Strategia: {pdf_stats.get('parent_child_count', 0)} Parent-Child, "
            f"{pdf_stats.get('direct_count', 0)} diretto"
        )

        routing = pdf_stats.get("routing", {})
        if routing:
            logger.info("     Routing breakdown:")
            for coll_name, count in routing.items():
                logger.info(f"       📁 {coll_name}: {count} documenti")

    except Exception as e:
        logger.error(f"  ❌ Errore indicizzazione PDF: {e}", exc_info=True)
        report["pdf_indexing"] = {"error": str(e)}

    # --- STEP 4: Verifica post-ingestion ---
    logger.info("\n[STEP 4/4] Verifica post-ingestion...")
    verification = verify_collections(indexer, settings)
    report["verification"] = verification

    log_sample_documents(indexer, max_per_collection=3)

    # --- REPORT FINALE ---
    report["duration_seconds"] = round(time.time() - start_time, 2)
    report["success"] = verification.get("ok", False) and _no_critical_errors(report)

    logger.info("\n" + "=" * 70)
    logger.info("📊 REPORT FINALE T6.4 — RE-INDICIZZAZIONE MULTI-COLLECTION")
    logger.info("=" * 70)
    logger.info(f"   Durata totale: {report['duration_seconds']}s")
    logger.info(f"   Chunks totali indicizzati: {verification.get('total_chunks', 0)}")
    logger.info(f"   Stato: {'✅ SUCCESSO' if report['success'] else '⚠️  CON PROBLEMI'}")

    for coll_name, count in verification.get("collections", {}).items():
        status = "✅" if count > 0 else "⚠️ "
        logger.info(f"   {status} {coll_name}: {count} chunks")

    logger.info("=" * 70)

    return report


# ============================================================
# HELPERS
# ============================================================

def _no_critical_errors(report: Dict[str, Any]) -> bool:
    """Controlla che non ci siano errori critici nel report."""
    for key in ("html_indexing", "pdf_indexing"):
        section = report.get(key)
        if isinstance(section, dict) and "error" in section:
            return False
    return True


def _load_html_cleaning_rules(settings: AppSettings):
    """
    Carica le regole di pulizia HTML dal modulo transform.

    NOTA: "obsolete_url" e "didattica" RIMOSSI dallo Sprint Filtri Docenti.
    """
    try:
        from transform.rules import get_all_html_rules
        rules = get_all_html_rules(
            directory=settings.ingestion.html_raw_dir,
            cutoff_year=settings.ingestion.cutoff_year,
            target_department=settings.ingestion.target_department,
        )
        logger.info(f"  Caricate {len(rules)} regole HTML cleaning")
        return rules
    except ImportError:
        logger.info("  Nessuna regola HTML cleaning trovata (modulo transform non disponibile)")
        return []


def _load_pdf_filter_rules(settings: AppSettings):
    """Carica le regole di filtro PDF dal modulo transform."""
    try:
        from transform.rules import get_all_pdf_rules
        rules = get_all_pdf_rules(cutoff_year=settings.ingestion.cutoff_year)
        logger.info(f"  Caricate {len(rules)} regole PDF filter")
        return rules
    except ImportError:
        logger.info("  Nessuna regola PDF filter trovata (modulo transform non disponibile)")
        return []


def save_report(report: Dict[str, Any], output_path: str = "t6_4_report.json") -> None:
    """Salva il report su file JSON."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"💾 Report salvato in: {output_path}")
    except Exception as e:
        logger.error(f"Errore salvataggio report: {e}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T6.4 — Re-indicizzazione completa multi-collection DIEM RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python ingestion_main.py                    # indicizza file esistenti (default)
  python ingestion_main.py --crawl            # crawling + indicizzazione
  python ingestion_main.py --verify-only      # solo verifica delle collection
  python ingestion_main.py --log-level DEBUG  # logging dettagliato
        """,
    )
    parser.add_argument(
        "--crawl",
        action="store_true",
        default=False,
        help="Esegui anche il crawling prima dell'indicizzazione.",
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        default=True,
        help="Salta il crawling e indicizza solo i file esistenti (default).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        default=False,
        help="Esegui solo la verifica delle collection.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="t6_4_report.json",
        help="Percorso del file di report JSON.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Livello di logging (default: INFO).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory di destinazione dei file di log.",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def setup_file_logger(log_level: str, log_dir: str = "logs") -> str:
    """Configura il logging su file .txt con nome timestampato."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"t6_4_ingestion_{timestamp_str}.txt"
    log_filepath = os.path.join(log_dir, log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(
        log_filepath, mode="w", encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    return log_filepath


def main() -> None:
    args = parse_args()

    log_filepath = setup_file_logger(args.log_level, log_dir=args.log_dir)
    print(f"📝 Log di esecuzione: {log_filepath}")

    settings = load_settings()

    if args.verify_only:
        print("🔎 Modalità VERIFY-ONLY: verifica delle collection senza indicizzazione")
        logger.info("🔎 Modalità VERIFY-ONLY")
        indexer = KnowledgeBaseIndexer(settings)
        verification = verify_collections(indexer, settings)
        log_sample_documents(indexer, max_per_collection=5)
        print(json.dumps(verification, indent=2, ensure_ascii=False))
        print(f"\n📝 Dettagli completi nel log: {log_filepath}")
        return

    print("🚀 Avvio ingestion T6.4... (l'output dettagliato è nel file di log)")
    skip_crawl = not args.crawl
    report = run_ingestion(settings, skip_crawl=skip_crawl)

    save_report(report, output_path=args.report_path)

    total_chunks = report.get("verification", {}).get("total_chunks", 0)
    duration = report.get("duration_seconds", 0)

    if not report.get("success"):
        logger.warning("⚠️  L'ingestion è terminata con problemi.")
        print(f"\n⚠️  Ingestion terminata CON PROBLEMI in {duration}s ({total_chunks} chunks)")
        print(f"   📝 Log completo: {log_filepath}")
        print(f"   📊 Report JSON: {args.report_path}")
        sys.exit(1)
    else:
        logger.info("✅ T6.4 completato con successo!")
        print(f"\n✅ T6.4 completato con SUCCESSO in {duration}s ({total_chunks} chunks)")
        print(f"   📝 Log completo: {log_filepath}")
        print(f"   📊 Report JSON: {args.report_path}")
        sys.exit(0)


if __name__ == "__main__":
    main()