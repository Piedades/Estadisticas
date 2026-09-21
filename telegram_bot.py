"""
Envío de avisos por Telegram (Bot API). No depende de que la app de
Streamlit esté sirviendo: recibe el token y el chat_id como parámetros, así
lo puede usar tanto la app (con los valores de st.secrets) como el script
standalone del GitHub Action programado (con variables de entorno).

Cómo conseguir el token y el chat_id (lo tiene que hacer Dani a mano, una
sola vez — crear el bot es un paso que Claude no puede hacer por él):
1. En Telegram, habla con @BotFather y usa el comando /newbot. Te da un
   token con forma "123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".
2. Escríbele cualquier mensaje a tu bot recién creado (para que sepa dónde
   contestarte).
3. Visita en el navegador:
   https://api.telegram.org/bot<TU_TOKEN>/getUpdates
   y busca "chat":{"id": ...} — ese número es tu chat_id.
4. Guarda ambos como secretos TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID: en
   Streamlit Cloud (Settings > Secrets, para el botón de prueba de la app)
   Y TAMBIÉN en GitHub (Settings > Secrets and variables > Actions, para
   el aviso diario automático) — son dos sitios distintos, hay que
   añadirlos en los dos.
"""
import requests

API_BASE = "https://api.telegram.org"


def send_message(token: str, chat_id: str, text: str):
    """Envía un mensaje de texto (admite HTML básico: <b>, <i>, saltos de
    línea). Devuelve (ok: bool, mensaje: str)."""
    if not token or not chat_id:
        return False, "Falta configurar TELEGRAM_BOT_TOKEN y/o TELEGRAM_CHAT_ID."
    try:
        resp = requests.post(
            f"{API_BASE}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=15,
        )
    except Exception as e:
        return False, f"Error de red enviando a Telegram: {e}"
    if resp.status_code == 200:
        return True, "Enviado correctamente."
    return False, f"Error {resp.status_code} de Telegram: {resp.text[:200]}"
