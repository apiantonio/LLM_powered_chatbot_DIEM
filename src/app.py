import streamlit as st
import sys
import os
import logging
import base64

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.settings import load_settings
from config.logging_config import setup_logging
from agent.agent_main import build_retrieval_engine, build_embedding_model, build_agent

logger = logging.getLogger(__name__)

LOGO_URL = "https://www.diem.unisa.it/rescue/img/headerbg/logo-diem.png"
FOOTER_URL = "https://www.diem.unisa.it/rescue/img/headerbg/footer-diem.png"

COLOR_PRIMARY = "#005A9C"
COLOR_BG_USER = "#BABFC2C3"

SUGGESTED_QUESTIONS = [
    "Cosa puoi fare per aiutarmi?",
    "Ho ottenuto un punteggio di 18 al TOLC-I. Posso iscrivermi?",
    "Dove si trova il DIEM e come posso raggiungerlo?",
]

@st.cache_resource(show_spinner=False)
def load_global_engines():
    settings = load_settings()
    
    setup_logging(
        level=settings.logging.level,
        log_file=settings.logging.log_file,
        log_to_console=settings.logging.log_to_console,
        log_format=settings.logging.log_format, 
        date_format=settings.logging.date_format,
    )
    logger.info("=== AVVIO APPLICAZIONE STREAMLIT DIEM CHATBOT ===")
    
    engine = build_retrieval_engine(settings)
    embedding_model = build_embedding_model(settings)
    return settings, engine, embedding_model

def init_session_agent():
    if "agent" not in st.session_state:
        boot_placeholder = st.empty()
        
        with boot_placeholder.status("Avvio dell'infrastruttura AI dipartimentale...", expanded=True) as status:
            st.write("Caricamento configurazioni...")
            settings, engine, embedding_model = load_global_engines()
            
            st.write("Inizializzazione sessione e memoria semantica...")
            st.session_state.agent = build_agent(
                settings=settings,
                engine=engine,
                enable_scope_guardrail=True,
                embedding_model=embedding_model,
            )
            status.update(label="Motore AI pronto!", state="complete", expanded=False)
            logger.info("Inizializzazione nuovo agente completata.")
            
        boot_placeholder.empty()
            
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "trace_mode" not in st.session_state:
        st.session_state.trace_mode = False
    if "active_prompt" not in st.session_state:
        st.session_state.active_prompt = None

def inject_custom_css():
    st.markdown(
        f"""
        <style>
        /* Typography globale */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;550;600&display=swap');
        body {{
            font-family: 'Poppins', -apple-system, sans-serif;
        }}

        /* Nascondiamo le ancore HTML e neutralizziamo il loro spazio */
        .user-msg-anchor, .bot-msg-anchor {{ 
            display: none !important; 
            position: absolute !important;
        }}

        /* =========================================
           LAYOUT CHAT: Separazione e allineamento
           ========================================= */
        [data-testid="stChatMessage"] {{
            display: flex !important;
            align-items: flex-start !important; /* Allineamento avatar-testo in alto */
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin-bottom: 2rem !important;
            width: 100% !important;
        }}

        /* Rimuove lo spazio vuoto extra sopra il primo paragrafo del testo markdown */
        [data-testid="stChatMessageContent"] p:first-child {{
            margin-top: 0 !important;
            padding-top: 0 !important;
        }}
        [data-testid="stStatusWidget"] {{
            display: none !important;
        }}

        /* === MESSAGGIO UTENTE (DESTRA) === */
        [data-testid="stChatMessage"]:has(.user-msg-anchor) {{
            flex-direction: row-reverse !important;
        }}
        [data-testid="stChatMessage"]:has(.user-msg-anchor) [data-testid="stChatMessageContent"] {{
            background-color: {COLOR_PRIMARY} !important;
            border-radius: 20px 0px 20px 20px !important;
            margin-left: auto !important; 
            margin-right: 15px !important; 
            padding: 10px 18px !important;
            max-width: 75% !important;
            width: fit-content !important;          
            flex: 0 1 auto !important;              
            overflow-wrap: break-word !important;   
            box-shadow: 0 4px 10px rgba(0, 90, 156, 0.15) !important;
        }}
        [data-testid="stChatMessage"]:has(.user-msg-anchor) [data-testid="stChatMessageContent"] * {{
            color: #FFFFFF !important;
        }}

        /* === MESSAGGIO BOT (SINISTRA - TESTO LIBERO) === */
        [data-testid="stChatMessage"]:has(.bot-msg-anchor) {{
            flex-direction: row !important;
        }}
        [data-testid="stChatMessage"]:has(.bot-msg-anchor) [data-testid="stChatMessageContent"] {{
            background-color: transparent !important; /* Rimosso riquadro */
            border-radius: 0 !important;
            margin-right: auto !important; 
            margin-left: 12px !important; 
            padding: 2px 0px 0px 0px !important; /* Nessun padding ai bordi, solo 2px in cima per allineare l'avatar */
            max-width: 85% !important;
            width: fit-content !important;          
            flex: 0 1 auto !important;
            overflow-wrap: break-word !important;
            box-shadow: none !important; /* Rimossa ombra */
            border: none !important; /* Rimosso bordo */
        }}

        /* =========================================
           FIX BARRA DI INPUT (Bordi Tondi & Focus)
           ========================================= */
        [data-testid="stChatInput"] {{
            background-color: transparent !important;
            padding-bottom: 15px !important;
        }}
        [data-testid="stChatInput"] textarea {{
            border: none !important;
            box-shadow: none !important;
        }}
        [data-testid="stChatInput"] > div {{
            border-radius: 25px !important; 
            overflow: hidden !important; 
            border: 1px solid #ccc !important;
            transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
        }}
        [data-testid="stChatInput"] > div:focus-within {{
            box-shadow: 0 0 0 2px {COLOR_PRIMARY} !important;
            border-color: {COLOR_PRIMARY} !important;
        }}
        /* Seleziona il cerchio di sfondo del pulsante quando è ATTIVO */
        [data-testid="stChatInput"] button {{
            background-color: {COLOR_PRIMARY} !important;
            border-color: {COLOR_PRIMARY} !important;
            color: #FFFFFF !important; /* Forza la freccia interna a essere bianca sul cerchio blu */
        }}

        /* Seleziona l'icona della freccia interna per assicurarsi che sia bianca */
        [data-testid="stChatInput"] button svg {{
            fill: #FFFFFF !important;
            color: #FFFFFF !important;
        }}

        /* Gestione dello stato HOVER (quando passi sopra con il mouse) */
        [data-testid="stChatInput"] button:hover {{
            background-color: #004070 !important; /* Un blu leggermente più scuro per dare feedback visivo */
            border-color: #004070 !important;
        }}

        /* Gestione dello stato DISABILITATO (quando la barra è vuota) */
        [data-testid="stChatInput"] button:disabled {{
            background-color: dimgray !important; /* Blu opaco/semitrasparente */
            border-color: transparent !important;
            opacity: 0.6 !important;
        }}
        [data-testid="stChatInput"] button:disabled svg {{
            fill: #ffffff !important;
            opacity: 0.5 !important;
        }}

        /* =========================================
           ANIMAZIONE SPINNER CUSTOM CON EMOJI
           ========================================= */
        .custom-spinner-wrapper {{
            display: flex;
            align-items: center;
            gap: 12px; /* Spazio ridotto */
            margin-bottom: 5px;
            transform: translateY(-14px); /* FIX: Sposta SOLO il blocco di caricamento in alto per allinearlo all'avatar */
        }}
        .custom-spinner-container {{
            position: relative;
            width: 24px;  /* Dimensioni ridotte */
            height: 24px; /* Dimensioni ridotte */
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .custom-spinner {{
            position: absolute;
            width: 100%;
            height: 100%;
            border: 3px solid #e0e0e0; /* Bordo più sottile */
            border-top: 3px solid {COLOR_PRIMARY};
            border-radius: 50%;
            animation: diem-spin 1s linear infinite;
        }}
        .custom-spinner-emoji {{
            font-size: 11px; /* Emoji più piccola */
            z-index: 1;
        }}
        .custom-spinner-text {{
            color: inherit;
            opacity: 0.8;
            font-style: italic;
            font-size: 0.9rem;
        }}
        @keyframes diem-spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* UX: Card per i bottoni suggeriti */
        .stButton > button {{
            border-radius: 16px;
            border: 1px solid rgba(0, 90, 156, 0.2);
            color: inherit;
            /* Sfondo colorato semi-trasparente: funziona sia col bianco che col nero! */
            background-color: rgba(0, 90, 156, 0.08) !important; 
            transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1);
            width: 100%;
            text-align: left;
            padding: 16px 20px;
            font-size: 0.95rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            font-weight: 500;
        }}
        .stButton > button:hover {{
            background-color: {COLOR_PRIMARY} !important;
            color: #FFFFFF !important;
            border: 1px solid {COLOR_PRIMARY};
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 90, 156, 0.15);
        }}
        
        /* Pannello laterale (Sidebar) */
        div[data-testid="stSidebar"] .stButton > button {{
            border: 1px solid #EAEAEA;
            color: inherit;
            background-color: transparent !important;
            border-radius: 10px;
            font-weight: 600;
            padding: 12px 16px;
            box-shadow: none;
            display: inline-flex !important;
            justify-content: center !important;
            align-items: center !important;
        }}
        div[data-testid="stSidebar"] .stButton > button:hover {{
            background-color: rgba(0, 0, 0, 0.05) !important;
            color: {COLOR_PRIMARY} !important;
            border-color: {COLOR_PRIMARY};
            transform: none; 
        }}
        .footer-img {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 75%;
            padding-top: 2rem;
            padding-bottom: 0.5rem;
            opacity: 0.9;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def handle_commands(command_type: str):
    if command_type == "reset":
        logger.info("Richiesto reset della memoria dall'utente.")
        st.session_state.agent.reset_memory()
        st.session_state.messages = []
        st.session_state.active_prompt = None
        st.rerun()
    elif command_type == "trace":
        st.session_state.trace_mode = not st.session_state.trace_mode
        status = "Visibili" if st.session_state.trace_mode else "Nascosti"
        logger.info(f"Modalità Trace: {status}.")
        st.toast(f"Dettagli di sistema: **{status}**", icon="⚙️")
    elif command_type == "quit":
        logger.info("Sessione terminata dall'utente.")
        st.session_state.agent.reset_memory()
        st.session_state.clear()
        st.success("Sessione scollegata in modo sicuro. Ricarica la pagina per un nuovo colloquio.")
        st.stop()

def format_latex(text: str) -> str:
    """Converte i delimitatori matematici del bot nel formato compatibile con Streamlit (KaTeX)."""
    if not text:
        return text
        
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    
    text = text.replace("[ \\text", "$$ \\text")
    text = text.replace("} ]", "} $$")
    
    return text

def render_sidebar():
    with st.sidebar:
        st.image(LOGO_URL, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Pannello di Controllo")
        st.caption("Gestisci le impostazioni della tua sessione.")
        
        if st.button("🔄 Pulisci Conversazione", help="Cancella la memoria recente e inizia un nuovo contesto", use_container_width=True):
            handle_commands("reset")
            
        trace_text = "🙈 Nascondi Metadati" if st.session_state.trace_mode else "🔍 Mostra Metadati (Trace)"
        if st.button(trace_text, help="Mostra log e retrieval (utile per il debugging)", use_container_width=True):
            handle_commands("trace")
            
        if st.button("🚪 Termina Sessione", help="Cancella tutti i dati e chiudi", use_container_width=True):
            handle_commands("quit")
        
        st.markdown("<br>"*3, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f'<img src="{FOOTER_URL}" class="footer-img">', unsafe_allow_html=True)
        st.caption("<center><small>© 2026 Dipartimento DIEM<br>Università degli Studi di Salerno</small></center>", unsafe_allow_html=True)

def get_image_base64(image_path: str) -> str:
    """Legge un'immagine locale e la converte in stringa Base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def render_chat_interface():
    col1, col2 = st.columns([1, 10])
    with col1:
        icon_path = "assets/bot.ico"
        try:
            icon_base64 = get_image_base64(icon_path)
            img_html = f'<img src="data:image/png;base64,{icon_base64}" style="width: 48px; height: 48px;">'
            st.markdown(f"<h1 style='text-align: center; margin-top: 0;'>{img_html}</h1>", unsafe_allow_html=True)
        except FileNotFoundError:
            logger.error(f"Impossibile trovare l'icona nel percorso: {icon_path}")
            st.markdown("<h1 style='text-align: center; margin-top: 0;'>🤖</h1>", unsafe_allow_html=True)
    with col2:
        st.title("Assistente Virtuale DIEM")
        
    st.markdown("*Benvenuto! Sono l'intelligenza artificiale a supporto di studenti e docenti del Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica Applicata.*")
    st.markdown("---")
    
    chat_input_prompt = st.chat_input("Scrivi qui la tua domanda o richiesta...")

    prompt = None
    if st.session_state.active_prompt is not None:
        prompt = st.session_state.active_prompt
        st.session_state.active_prompt = None
    elif chat_input_prompt:
        logger.info(f"Input utente tramite chat box: '{chat_input_prompt}'")
        prompt = chat_input_prompt

    for msg in st.session_state.messages:
        avatar = "assets/student.ico" if msg["role"] == "user" else "assets/chat.ico"
        with st.chat_message(msg["role"], avatar=avatar):
            anchor = "<span class='user-msg-anchor'></span>" if msg["role"] == "user" else "<span class='bot-msg-anchor'></span>"
            content_to_render = format_latex(msg['content']) if msg["role"] == "assistant" else msg['content']
            st.markdown(f"{anchor}{content_to_render}", unsafe_allow_html=True)
            
            if st.session_state.trace_mode and msg["role"] == "assistant" and "trace" in msg:
                with st.expander("🛠️ Dati di Tracciamento e Retrieval", expanded=False):
                    st.json(msg["trace"])

    suggestions_placeholder = st.empty()
    if len(st.session_state.messages) == 0 and not prompt:
        with suggestions_placeholder.container():
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Prova con una di queste domande:")
            
            cols = st.columns(3)
            for i, question in enumerate(SUGGESTED_QUESTIONS):
                with cols[i % 3]:
                    if st.button(question, key=f"sug_btn_{i}"):
                        logger.info(f"Input utente tramite card UI: '{question}'")
                        st.session_state.active_prompt = question
                        suggestions_placeholder.empty() 
                        st.rerun()

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="assets/student.ico"):
            st.markdown(f"<span class='user-msg-anchor'></span>{prompt}", unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="assets/chat.ico"):
            loader_placeholder = st.empty()
            loader_placeholder.markdown("""
                <span class='bot-msg-anchor'></span>
                <div class="custom-spinner-wrapper">
                    <div class="custom-spinner-container">
                        <div class="custom-spinner"></div>
                        <div class="custom-spinner-emoji">🔎</div>
                    </div>
                    <div class="custom-spinner-text">Ricerca delle informazioni...</div>
                </div>
            """, unsafe_allow_html=True)
            
            try:
                logger.debug(f"Chiamata a agent.chat() per: '{prompt}'")
                result_dict = st.session_state.agent.chat(prompt)
                
                loader_placeholder.empty()
    
                raw_response = result_dict.get("response", "Nessuna risposta dal sistema.")
                response_text = format_latex(raw_response)
                
                trace_data = result_dict.get("trace", {})
                is_blocked = result_dict.get("blocked", False)
                
                if is_blocked:
                    st.error("🛑 " + response_text)
                    logger.warning(f"Blocco guardrail: {response_text}")
                else:
                    st.markdown(f"<span class='bot-msg-anchor'></span>{response_text}", unsafe_allow_html=True)
                    
                if st.session_state.trace_mode:
                    with st.expander("🛠️ Dati di Tracciamento e Retrieval", expanded=False):
                        st.json(trace_data)
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "trace": trace_data
                })
            except Exception as e:
                loader_placeholder.empty()
                logger.error(f"Eccezione Runtime nell'Agente: {str(e)}", exc_info=True)
                st.error("⚠️ Si è verificato un errore tecnico durante l'elaborazione. I log di sistema sono stati aggiornati.")


def main():
    st.set_page_config(
        page_title="Chatbot DIEM",
        page_icon="assets/bot.ico",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    init_session_agent()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()