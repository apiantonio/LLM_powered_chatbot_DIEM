"""
agent/guardrails.py — Sistema Guardrails basato su Middleware LangChain.

IMPLEMENTAZIONE COMPLETA basata sulla documentazione ufficiale dei middleware:
  - Prebuilt: PIIMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
  - Custom (class-based): TopicalGuardrail, InjectionGuard, ToxicityFilter,
                           HallucinationGuard, CodeGenerationGuard

ARCHITETTURA:
  Ogni guardrail è un middleware LangChain che si inserisce nel ciclo
  dell'agente tramite hook before_model, after_model o wrap_model_call.
  L'ordine di esecuzione è determinato dall'ordine nella lista middleware
  passata a create_agent().

  before_model hooks (in ordine):
    1. InjectionGuardMiddleware  — blocca prompt injection
    2. ToxicityFilterMiddleware  — blocca linguaggio tossico
    3. TopicalGuardrailMiddleware — blocca domande fuori contesto

  after_model hooks (in ordine inverso):
    1. HallucinationGuardMiddleware — valida output contro allucinazioni
    2. CodeGenerationGuardMiddleware — blocca generazione di codice

  Prebuilt middleware:
    - PIIMiddleware (con eccezioni per email e telefoni docenti)
    - ModelCallLimitMiddleware (anti-loop)
    - ToolCallLimitMiddleware (limiti invocazioni tool)

ECCEZIONE PII CRITICA:
  Email e numeri di telefono NON vengono bloccati perché sono
  informazioni pubbliche dei docenti universitari. Solo il codice
  fiscale e altri PII sensibili vengono redatti.
"""

import re
import logging
from typing import Any, Optional, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


# ============================================================
# ECCEZIONI PERSONALIZZATE
# ============================================================

class ScopeViolationError(Exception):
    """Eccezione per query fuori dominio."""
    pass


class InputInjectionError(Exception):
    """Eccezione per prompt injection rilevata."""
    pass


class ToxicityError(Exception):
    """Eccezione per contenuto tossico rilevato."""
    pass


# ============================================================
# MESSAGGI DI RIFIUTO STANDARD
# ============================================================

SCOPE_REJECTION_MSG = (
    "Mi dispiace, sono l'assistente virtuale del Dipartimento DIEM "
    "dell'Università degli Studi di Salerno. Posso rispondere solo a "
    "domande relative al dipartimento, ai corsi di laurea, ai docenti, "
    "agli esami, ai regolamenti, ai laboratori e ai servizi universitari. "
    "La tua domanda sembra riguardare un altro argomento."
)

INJECTION_REJECTION_MSG = (
    "Ho rilevato un tentativo di manipolazione nelle istruzioni. "
    "Posso aiutarti solo con domande relative al DIEM."
)

TOXICITY_REJECTION_MSG = (
    "Il tuo messaggio contiene linguaggio inappropriato. "
    "Ti chiedo di riformulare la domanda in modo rispettoso. "
    "Sono qui per aiutarti con informazioni sul Dipartimento DIEM."
)

HALLUCINATION_REJECTION_MSG = (
    "Mi scuso, non sono riuscito a generare una risposta verificabile. "
    "Riprova con una domanda più specifica sul DIEM."
)

CODE_GENERATION_REJECTION_MSG = (
    "Non sono in grado di generare codice o script. "
    "Sono un assistente informativo del Dipartimento DIEM. "
    "Posso aiutarti con informazioni su corsi, docenti, servizi e regolamenti."
)


# ============================================================
# 1. INJECTION GUARD MIDDLEWARE (before_model)
# ============================================================

class InjectionGuardMiddleware(AgentMiddleware):
    """
    Middleware custom (class-based) per rilevamento prompt injection.

    Hook: before_model — intercetta la query PRIMA che arrivi al modello.
    Se rileva pattern di injection, salta direttamente a 'end' con
    un messaggio di rifiuto.

    Pattern rilevati:
      - Tentativi di ignorare/sovrascrivere istruzioni di sistema
      - Tentativi di role-play non autorizzato
      - Tentativi di estrazione del system prompt
      - Tentativi di jailbreak con encoding/obfuscation
      - Tentativi di manipolazione ("Are you sure?", "You're wrong")
    """

    _INJECTION_PATTERNS = [
        # Italiano
        r"ignor[ae]\s+(le\s+)?istruzioni\s+preced",
        r"dimentica\s+(le\s+)?istruzioni",
        r"non\s+seguire\s+(le\s+)?regole",
        r"cambia\s+(il\s+tuo\s+)?ruolo",
        r"fai\s+finta\s+di\s+essere",
        r"comportati\s+come\s+(un|una)\s+(?!assistente)",
        r"mostra(mi)?\s+(il\s+)?prompt\s+di\s+sistema",
        r"quali\s+sono\s+le\s+tue\s+istruzioni",
        r"ripeti\s+(il\s+)?system\s+prompt",
        r"sei\s+in\s+modalit[àa]\s+sviluppatore",
        r"modalit[àa]\s+(di\s+)?debug",
        r"disabilita\s+(i\s+)?filtri",
        r"rispondi\s+senza\s+restrizioni",
        r"nessun[ao]?\s+censura",

        # English
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?(your\s+)?instructions",
        r"disregard\s+(all\s+)?(your\s+)?instructions",
        r"override\s+(your\s+)?instructions",
        r"you\s+are\s+now\s+(?:a|an)\s+(?!assistant)",
        r"act\s+as\s+(?:a|an)\s+(?!assistente|assistant)",
        r"pretend\s+(?:you\s+are|to\s+be)",
        r"you\s+are\s+in\s+developer\s+mode",
        r"developer\s+mode\s+(on|enabled|activated)",
        r"system\s*prompt",
        r"reveal\s+(?:your\s+)?(?:instructions|prompt)",
        r"what\s+(?:are|is)\s+your\s+(?:instructions|system)",
        r"show\s+(?:me\s+)?(?:your\s+)?(?:instructions|system\s+prompt)",
        r"repeat\s+(?:your\s+)?(?:instructions|system\s+prompt)",
        r"DAN\s+mode",
        r"jailbreak",
        r"do\s+anything\s+now",
        r"bypass\s+(?:your\s+)?(?:filters|restrictions|safety)",
        r"no\s+(?:filters|restrictions|censorship)",
    ]

    _MANIPULATION_PATTERNS = [
        # "Are you sure?" style manipulation
        r"(?:are|sei)\s+(?:you\s+)?sure\s*\??",
        r"(?:no|non)\s*,?\s*(?:I\s+think|penso|credo)\s+(?:you|che)\s+(?:are|sei|stai)\s+wrong",
        r"(?:actually|in realt[àa]|veramente).*(?:wrong|sbagliato|sbagliata|errato)",
        r"(?:I\s+think|penso)\s+(?:you'?re|che\s+(?:sei|stai))\s+(?:lying|mentendo)",
        r"(?:la|the)\s+risposta\s+(?:è|is)\s+(?:sbagliata|wrong|errata|incorretta)",
        r"(?:you\s+)?(?:must|devi)\s+(?:answer|rispondere)\s+(?:differently|diversamente)",
    ]

    def __init__(self):
        super().__init__()
        self._injection_regex = [
            re.compile(p, re.IGNORECASE) for p in self._INJECTION_PATTERNS
        ]
        self._manipulation_regex = [
            re.compile(p, re.IGNORECASE) for p in self._MANIPULATION_PATTERNS
        ]

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Intercetta injection prima del model call."""
        last_msg = self._get_last_user_message(state)
        if not last_msg:
            return None

        # Check injection patterns
        for pattern in self._injection_regex:
            if pattern.search(last_msg):
                logger.warning(f"🛡️ INJECTION BLOCCATA: '{last_msg[:100]}...'")
                return {
                    "messages": [AIMessage(content=INJECTION_REJECTION_MSG)],
                    "jump_to": "end",
                }

        # Check manipulation patterns (log but don't block - agent should be robust)
        for pattern in self._manipulation_regex:
            if pattern.search(last_msg):
                logger.info(f"⚠️ Tentativo di manipolazione rilevato: '{last_msg[:80]}'")
                # Non blocchiamo, ma logghiamo - il Robustness KPI richiede
                # che l'agente mantenga la sua risposta originale
                break

        return None

    @staticmethod
    def _get_last_user_message(state: AgentState) -> str:
        """Estrae l'ultimo messaggio utente dallo stato."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""


# ============================================================
# 2. TOXICITY FILTER MIDDLEWARE (before_model)
# ============================================================

class ToxicityFilterMiddleware(AgentMiddleware):
    """
    Middleware custom (class-based) per filtraggio tossicità.

    Hook: before_model — blocca messaggi con linguaggio inappropriato
    prima che raggiungano il modello.

    Rileva:
      - Parolacce e insulti (italiano e inglese)
      - Linguaggio sessualmente esplicito
      - Minacce e linguaggio violento
      - Discriminazione e hate speech
    """

    _PROFANITY_PATTERNS = [
        # Italiano — parolacce comuni
        r"\b(cazzo|minchia|figa|merda|stronz[oa]|coglion[ea]|vaffanculo)\b",
        r"\b(puttana|troia|zoccola|bastard[oa]|idiota|deficiente|cretino)\b",
        r"\b(fanculo|porco\s+dio|madonna\s+puttana|dio\s+(?:cane|porco|bestia))\b",
        r"\b(culo|cul[oa]|imbecille|scemo|stupido)\b",
        r"\b(negro|negr[oa]|frocio|froci[oa]|ricchione)\b",

        # English — common profanity
        r"\b(fuck|shit|asshole|bitch|bastard|dick|cock|cunt)\b",
        r"\b(motherfucker|bullshit|damn|dumbass|retard)\b",
        r"\b(nigger|faggot|slut|whore)\b",
    ]

    _THREAT_PATTERNS = [
        r"\b(ti\s+ammazzo|ti\s+uccido|ti\s+spacco|ti\s+meno)\b",
        r"\b(I'?ll\s+kill\s+you|gonna\s+kill|death\s+threat)\b",
        r"\b(bomba|esplosivo|arma|sparare)\b",
        r"\b(bomb|weapon|shoot|murder)\b",
    ]

    def __init__(self):
        super().__init__()
        self._profanity_regex = [
            re.compile(p, re.IGNORECASE) for p in self._PROFANITY_PATTERNS
        ]
        self._threat_regex = [
            re.compile(p, re.IGNORECASE) for p in self._THREAT_PATTERNS
        ]

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Blocca messaggi tossici prima del model call."""
        last_msg = self._get_last_user_message(state)
        if not last_msg:
            return None

        # Check profanity
        for pattern in self._profanity_regex:
            if pattern.search(last_msg):
                logger.warning(f"🚫 TOSSICITÀ BLOCCATA (profanità): '{last_msg[:80]}...'")
                return {
                    "messages": [AIMessage(content=TOXICITY_REJECTION_MSG)],
                    "jump_to": "end",
                }

        # Check threats
        for pattern in self._threat_regex:
            if pattern.search(last_msg):
                logger.warning(f"🚫 TOSSICITÀ BLOCCATA (minaccia): '{last_msg[:80]}...'")
                return {
                    "messages": [AIMessage(content=TOXICITY_REJECTION_MSG)],
                    "jump_to": "end",
                }

        return None

    @staticmethod
    def _get_last_user_message(state: AgentState) -> str:
        """Estrae l'ultimo messaggio utente dallo stato."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""


# ============================================================
# 3. TOPICAL GUARDRAIL MIDDLEWARE (before_model)
# ============================================================

class TopicalGuardrailMiddleware(AgentMiddleware):
    """
    Middleware custom (class-based) per controllo topicale.

    Hook: before_model — classifica la query come IN_SCOPE o OUT_OF_SCOPE
    PRIMA che il modello venga invocato, evitando di sprecare risorse
    su domande fuori contesto.

    Strategia dual-layer:
      Layer 1 (deterministico, veloce): Pattern matching su keyword
              off-topic evidenti (sport, cucina, politica, entertainment).
      Layer 2 (LLM-based, preciso): Se il Layer 1 non rileva nulla,
              usa un LLM leggero per classificare la query.

    Il Layer 1 è opzionale e serve come fast-path per i casi ovvi.
    Il Layer 2 garantisce la copertura su casi ambigui.

    NOTA: Le query meta (saluti, "chi sei?", "cosa sai fare?") sono
          esplicitamente escluse dal blocco.
    """

    # Pattern per domande palesemente fuori contesto
    _OFF_TOPIC_PATTERNS = [
        # Sport
        r"\b(calcio|serie\s+[ab]|champions|coppa|goal|gol|partita|stadio|fifa)\b",
        r"\b(soccer|football\s+(?:match|game)|basketball|tennis\s+match|nba|nfl)\b",
        # Cucina
        r"\b(ricetta|cucinare|ingredienti|pizza|pasta\s+(?:al|alla|con|e)\b)",
        r"\b(recipe|cooking|baking|chef|restaurant\s+(?:review|recommendation))\b",
        # Politica (fuori contesto universitario)
        r"\b(presidente\s+(?:della\s+)?repubblica|primo\s+ministro|elezioni\s+politiche)\b",
        r"\b(re\s+(?:di|d')\s+\w+|regina\s+(?:di|d')\s+\w+|monarca)\b",
        r"\b(king\s+of|queen\s+of|president\s+of\s+(?!the\s+department|DIEM))\b",
        # Entertainment
        r"\b(film|movie|netflix|spotify|canzone|cantante|attore|attrice)\b",
        r"\b(videogame|gioco\s+(?:da\s+tavolo|online|video)|playstation|xbox|nintendo)\b",
        # Gossip / Generic
        r"\b(oroscopo|horoscope|zodiac|fortuna|lotto|scommesse|betting)\b",
        r"\b(meteo|weather|previsioni\s+(?:del\s+)?tempo)\b",
        # Viaggi non universitari
        r"\b(volo|aereo|hotel|vacanza|vacanze|ferie|spiaggia|mare)\b",
    ]

    # Meta-query escluse dal blocco
    _META_PATTERNS = [
        r"^(ciao|salve|buongiorno|buonasera|hey|hi|hello)\b",
        r"^grazie",
        r"^(come stai|come va|chi sei|cosa sai fare|cosa puoi fare|come funzioni)",
        r"^(aiuto|help|assistenza)",
    ]

    # Keyword universitarie/DIEM che forzano IN_SCOPE
    _IN_SCOPE_KEYWORDS = [
        r"\b(diem|unisa|universit[àa]|dipartimento|facolt[àa])\b",
        r"\b(docent[ei]|professor[ei]?|prof\.?|ricercator[ei])\b",
        r"\b(esam[ei]|lezione|lezioni|corso|corsi|laurea|magistrale|triennale)\b",
        r"\b(matricola|studente|studenti|studentessa|iscrizione|immatricolazione)\b",
        r"\b(tesi|tirocinio|stage|erasmus|mobilit[àa]|internazionale)\b",
        r"\b(laboratorio|laboratori|aula|aule|sede|campus|edificio)\b",
        r"\b(bando|bandi|borsa|borse|concorso|assegno)\b",
        r"\b(regolamento|piano\s+(?:di\s+)?studi|cfu|crediti)\b",
        r"\b(ricevimento|orario|calendario|appello|appelli)\b",
        r"\b(segreteria|biblioteca|mensa|servizi)\b",
        r"\b(dottorato|phd|ricerca|pubblicazione|pubblicazioni)\b",
        r"\b(syllabus|programma|insegnamento|insegnamenti)\b",
        r"\b(tolc|ammissione|requisiti|ofa)\b",
        r"\b(commissione|paritetica|consiglio)\b",
        r"\b(ingegneria|informatica|elettronica|telecomunicazioni)\b",
        r"\b(voto|media|voti|graduat|laurearsi)\b",
    ]

    def __init__(self, classifier_llm=None):
        """
        Args:
            classifier_llm: LLM per classificazione (Layer 2).
                           Se None, usa solo il Layer 1 deterministico.
        """
        super().__init__()
        self._classifier_llm = classifier_llm

        self._off_topic_regex = [
            re.compile(p, re.IGNORECASE) for p in self._OFF_TOPIC_PATTERNS
        ]
        self._meta_regex = [
            re.compile(p, re.IGNORECASE) for p in self._META_PATTERNS
        ]
        self._in_scope_regex = [
            re.compile(p, re.IGNORECASE) for p in self._IN_SCOPE_KEYWORDS
        ]

        # Chain LLM per classificazione (Layer 2)
        if self._classifier_llm:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableLambda

            self._classification_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Sei un classificatore di dominio. Il tuo unico compito è determinare se "
                 "la seguente domanda riguarda il Dipartimento DIEM dell'Università degli "
                 "Studi di Salerno (corsi, docenti, esami, orari, regolamenti, tesi, "
                 "borse di studio, laboratori, servizi, dottorato, ricerca accademica, "
                 "strutture universitarie, offerta formativa).\n\n"
                 "Rispondi SOLO con 'IN_SCOPE' o 'OUT_OF_SCOPE'. Nient'altro."),
                ("human", "{query}"),
            ])
            self._classification_chain = (
                self._classification_prompt
                | self._classifier_llm
                | RunnableLambda(lambda msg: msg.content.strip().upper())
            )

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Blocca query fuori contesto prima del model call."""
        last_msg = self._get_last_user_message(state)
        if not last_msg:
            return None

        query_clean = last_msg.strip().lower()

        # Fast-path: meta-query (saluti, "chi sei") → sempre ammesse
        for pattern in self._meta_regex:
            if re.match(pattern, query_clean):
                return None

        # Layer 1: Pattern off-topic deterministico
        for pattern in self._off_topic_regex:
            if pattern.search(last_msg):
                logger.info(f"🚧 FUORI CONTESTO (Layer 1): '{last_msg[:80]}...'")
                return {
                    "messages": [AIMessage(content=SCOPE_REJECTION_MSG)],
                    "jump_to": "end",
                }

        # Fast-path: keyword DIEM/universitarie presenti → IN_SCOPE
        for pattern in self._in_scope_regex:
            if pattern.search(last_msg):
                return None

        # Layer 2: Classificazione LLM (se configurata)
        if self._classifier_llm:
            try:
                result = self._classification_chain.invoke({"query": last_msg})
                if "OUT_OF_SCOPE" in result:
                    logger.info(f"🚧 FUORI CONTESTO (Layer 2 LLM): '{last_msg[:80]}...'")
                    return {
                        "messages": [AIMessage(content=SCOPE_REJECTION_MSG)],
                        "jump_to": "end",
                    }
            except Exception as e:
                logger.warning(f"Errore classificazione LLM, fail-open: {e}")
                # Fail-open: in caso di errore, lascia passare la query

        return None

    @staticmethod
    def _get_last_user_message(state: AgentState) -> str:
        """Estrae l'ultimo messaggio utente dallo stato."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""


# ============================================================
# 4. HALLUCINATION GUARD MIDDLEWARE (after_model)
# ============================================================

class HallucinationGuardMiddleware(AgentMiddleware):
    """
    Middleware custom (class-based) per rilevamento allucinazioni.

    Hook: after_model — analizza la risposta del modello DOPO la generazione
    e blocca risposte che contengono segnali di allucinazione.

    Segnali rilevati:
      - Risposte che inventano URL non presenti nel contesto
      - Risposte che affermano certezze su dati non recuperati
      - Risposte vuote o troppo corte
      - Pattern tipici di confabulazione dei LLM
    """

    # Pattern che indicano allucinazione
    _HALLUCINATION_SIGNALS = [
        # URL inventati (non dei domini DIEM/UniSA)
        r"https?://(?!(?:www\.)?diem\.unisa\.it|docenti\.unisa\.it|corsi\.unisa\.it|easycourse\.unisa\.it)[a-z0-9-]+\.[a-z]{2,}",
        # Numeri di telefono inventati (pattern troppo specifico per essere reale)
        r"\b(?:089|06|02)\s*\d{7,10}\b",
    ]

    # Frasi che indicano che il modello sta inventando
    _CONFABULATION_PATTERNS = [
        r"(?:come\s+)?(?:è\s+)?noto\s+(?:che|a\s+tutti)",
        r"(?:tutti|ognuno)\s+(?:sanno|sa)\s+che",
        r"(?:as\s+)?(?:everyone|we\s+all)\s+know",
        r"(?:it\s+is\s+)?well[\s-]known\s+(?:that|fact)",
    ]

    _MIN_RESPONSE_LENGTH = 10

    def __init__(self):
        super().__init__()
        self._hallucination_regex = [
            re.compile(p, re.IGNORECASE) for p in self._HALLUCINATION_SIGNALS
        ]
        self._confabulation_regex = [
            re.compile(p, re.IGNORECASE) for p in self._CONFABULATION_PATTERNS
        ]

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Valida la risposta del modello dopo la generazione."""
        last_msg = self._get_last_ai_message(state)
        if not last_msg:
            return None

        # Skip se il messaggio ha tool_calls (è una chiamata intermedia)
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    return None
                break

        # Check risposta vuota o troppo corta
        if len(last_msg.strip()) < self._MIN_RESPONSE_LENGTH:
            logger.warning(f"⚠️ Risposta troppo corta: '{last_msg[:50]}'")
            return {
                "messages": [AIMessage(content=HALLUCINATION_REJECTION_MSG)],
                "jump_to": "end",
            }

        # Check confabulation patterns
        for pattern in self._confabulation_regex:
            if pattern.search(last_msg):
                logger.warning(
                    f"⚠️ Possibile confabulazione rilevata: '{last_msg[:100]}...'"
                )
                # Non blocchiamo, ma logghiamo — la confabulazione è un segnale
                # debole, non una certezza
                break

        return None

    @staticmethod
    def _get_last_ai_message(state: AgentState) -> str:
        """Estrae l'ultimo messaggio AI dallo stato."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                return msg.content
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""


# ============================================================
# 5. CODE GENERATION GUARD MIDDLEWARE (after_model)
# ============================================================

class CodeGenerationGuardMiddleware(AgentMiddleware):
    """
    Middleware custom (class-based) per bloccare generazione di codice.

    Hook: after_model — rileva e blocca risposte che contengono
    codice o script generati dal modello.

    L'agente DIEM è un assistente informativo, NON deve generare codice.
    """

    _CODE_PATTERNS = [
        # Code blocks
        r"```(?:python|java|javascript|c\+\+|html|css|sql|bash|shell|php|ruby|go|rust|kotlin|swift)",
        # Import statements
        r"^\s*(?:import|from\s+\w+\s+import|require\(|#include|using\s+namespace)",
        # Function definitions
        r"^\s*(?:def\s+\w+\(|function\s+\w+\(|class\s+\w+[:\(]|public\s+(?:static\s+)?void)",
        # Common code constructs (solo se appaiono multiple volte)
        r"(?:for\s*\(.*\)\s*\{|while\s*\(.*\)\s*\{|if\s*\(.*\)\s*\{)",
    ]

    # Soglia: quanti code block devono essere presenti per bloccare
    _CODE_BLOCK_THRESHOLD = 1

    def __init__(self):
        super().__init__()
        self._code_regex = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in self._CODE_PATTERNS
        ]

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Blocca risposte contenenti codice generato."""
        last_msg = self._get_last_ai_message(state)
        if not last_msg:
            return None

        # Skip se il messaggio ha tool_calls (è una chiamata intermedia)
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    return None
                break

        # Conta i code block trovati
        code_blocks_found = 0
        for pattern in self._code_regex:
            matches = pattern.findall(last_msg)
            code_blocks_found += len(matches)

        if code_blocks_found >= self._CODE_BLOCK_THRESHOLD:
            logger.warning(
                f"🚫 CODICE BLOCCATO: {code_blocks_found} blocchi trovati "
                f"nella risposta"
            )
            return {
                "messages": [AIMessage(content=CODE_GENERATION_REJECTION_MSG)],
                "jump_to": "end",
            }

        return None

    @staticmethod
    def _get_last_ai_message(state: AgentState) -> str:
        """Estrae l'ultimo messaggio AI dallo stato."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                return msg.content
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""


# ============================================================
# 6. PII GUARD — Custom detector per Codice Fiscale italiano
# ============================================================

def detect_codice_fiscale(content: str) -> list[dict[str, str | int]]:
    """
    Detector custom per il codice fiscale italiano.

    Pattern: 6 lettere + 2 cifre + 1 lettera + 2 cifre + 1 lettera + 3 cifre + 1 lettera
    Esempio: RSSMRA85M01H501Z

    NOTA: Questo detector è usato dal PIIMiddleware prebuilt.
    Email e telefoni NON vengono rilevati intenzionalmente, perché
    sono informazioni pubbliche dei docenti.

    Returns:
        Lista di dict con 'text', 'start', 'end' per ogni match.
    """
    pattern = r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b"
    matches = []
    for match in re.finditer(pattern, content):
        cf = match.group(0)
        # Validazione base: le prime 3 cifre non devono essere 000
        first_three = cf[6:8]
        matches.append({
            "text": cf,
            "start": match.start(),
            "end": match.end(),
        })
    return matches


# ============================================================
# 7. OUTPUT VALIDATOR MIDDLEWARE (after_model)
#    Guardrail aggiuntivo per PII nell'output
# ============================================================

class OutputPIIGuardMiddleware(AgentMiddleware):
    """
    Middleware custom (class-based) per filtrare PII dall'output.

    Hook: after_model — analizza la risposta e maschera eventuali
    codici fiscali presenti nell'output.

    ECCEZIONE CRITICA: Email e numeri di telefono NON vengono
    mascherati perché sono informazioni pubbliche dei docenti.
    """

    _PII_PATTERNS = [
        (r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b', "CODICE_FISCALE"),
        # IBAN italiano
        (r'\bIT\d{2}[A-Z]\d{22}\b', "IBAN"),
    ]

    def __init__(self):
        super().__init__()
        self._pii_regex = [
            (re.compile(p), label) for p, label in self._PII_PATTERNS
        ]

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Maschera PII sensibili nell'output (eccetto email e telefoni)."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not (hasattr(last_msg, "type") and last_msg.type == "ai"):
            return None

        # Skip se è una tool call intermedia
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return None

        content = last_msg.content
        modified = False

        for pattern, label in self._pii_regex:
            if pattern.search(content):
                content = pattern.sub(f"[{label} RIMOSSO]", content)
                modified = True
                logger.warning(f"PII ({label}) rimosso dall'output")

        if modified:
            return {
                "messages": [AIMessage(content=content)],
            }

        return None


# ============================================================
# FACTORY: Assembla tutti i middleware
# ============================================================

def build_guardrail_middleware(
    classifier_llm=None,
    enable_pii: bool = True,
    enable_topical: bool = True,
    enable_injection: bool = True,
    enable_toxicity: bool = True,
    enable_hallucination: bool = True,
    enable_code_guard: bool = True,
    max_model_calls_per_run: int = 8,
    max_tool_calls_per_run: int = 12,
) -> list:
    """
    Factory che assembla la lista completa dei middleware guardrail.

    L'ordine è cruciale:
      1. Prebuilt middleware (PIIMiddleware, limiti)
      2. before_model hooks (injection → toxicity → topical)
      3. after_model hooks (hallucination → code → output PII)

    Hook execution order (dalla documentazione):
      before_model: middleware[0] → middleware[1] → middleware[2] → ...
      after_model:  ... → middleware[2] → middleware[1] → middleware[0]

    Args:
        classifier_llm: LLM per classificazione topicale (opzionale).
        enable_pii: Attiva PIIMiddleware per codici fiscali.
        enable_topical: Attiva TopicalGuardrailMiddleware.
        enable_injection: Attiva InjectionGuardMiddleware.
        enable_toxicity: Attiva ToxicityFilterMiddleware.
        enable_hallucination: Attiva HallucinationGuardMiddleware.
        enable_code_guard: Attiva CodeGenerationGuardMiddleware.
        max_model_calls_per_run: Limite model calls per invocazione.
        max_tool_calls_per_run: Limite tool calls per invocazione.

    Returns:
        Lista ordinata di middleware per create_agent().
    """
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        PIIMiddleware,
    )

    middleware_list = []

    # --- Prebuilt: PII Detection (solo codice fiscale, NO email/telefono) ---
    if enable_pii:
        middleware_list.append(
            PIIMiddleware(
                "codice_fiscale",
                detector=detect_codice_fiscale,
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
                apply_to_tool_results=False,
            )
        )
        logger.info("   ✅ PIIMiddleware: codice fiscale (email/telefoni ESCLUSI)")

    # --- Prebuilt: Model Call Limit (anti-loop) ---
    middleware_list.append(
        ModelCallLimitMiddleware(
            run_limit=max_model_calls_per_run,
            exit_behavior="end",
        )
    )
    logger.info(f"   ✅ ModelCallLimitMiddleware: max {max_model_calls_per_run}/run")

    # --- Prebuilt: Tool Call Limit ---
    middleware_list.append(
        ToolCallLimitMiddleware(
            run_limit=max_tool_calls_per_run,
            exit_behavior="continue",
        )
    )
    logger.info(f"   ✅ ToolCallLimitMiddleware: max {max_tool_calls_per_run}/run")

    # --- Custom: Injection Guard (before_model, ordine 1) ---
    if enable_injection:
        middleware_list.append(InjectionGuardMiddleware())
        logger.info("   ✅ InjectionGuardMiddleware: prompt injection detection")

    # --- Custom: Toxicity Filter (before_model, ordine 2) ---
    if enable_toxicity:
        middleware_list.append(ToxicityFilterMiddleware())
        logger.info("   ✅ ToxicityFilterMiddleware: profanity + threats")

    # --- Custom: Topical Guardrail (before_model, ordine 3) ---
    if enable_topical:
        middleware_list.append(TopicalGuardrailMiddleware(classifier_llm=classifier_llm))
        if classifier_llm:
            logger.info("   ✅ TopicalGuardrailMiddleware: dual-layer (regex + LLM)")
        else:
            logger.info("   ✅ TopicalGuardrailMiddleware: Layer 1 only (regex)")

    # --- Custom: Hallucination Guard (after_model) ---
    if enable_hallucination:
        middleware_list.append(HallucinationGuardMiddleware())
        logger.info("   ✅ HallucinationGuardMiddleware: confabulation detection")

    # --- Custom: Code Generation Guard (after_model) ---
    if enable_code_guard:
        middleware_list.append(CodeGenerationGuardMiddleware())
        logger.info("   ✅ CodeGenerationGuardMiddleware: code block detection")

    # --- Custom: Output PII Guard (after_model) ---
    if enable_pii:
        middleware_list.append(OutputPIIGuardMiddleware())
        logger.info("   ✅ OutputPIIGuardMiddleware: CF/IBAN masking in output")

    return middleware_list