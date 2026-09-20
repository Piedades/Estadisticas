"""
Actualización automática de datos desde football-data.co.uk: descarga el
CSV de la temporada en curso para cada liga y lo sube a GitHub, sin tener
que subir nada a mano jornada tras jornada.

football-data.co.uk publica un único CSV acumulativo por temporada y liga
(se va ampliando partido a partido), así que basta con volver a descargarlo
y sobreescribir siempre el mismo archivo para tener los datos al día.
"""
from datetime import date

import requests

# Import diferido a propósito (dentro de update_league, no aquí arriba):
# github_sync importa streamlit, que no está disponible cuando este módulo
# se usa desde el script standalone de GitHub Actions (scripts/update_data_ci.py),
# que solo necesita current_season_code()/fetch_league_csv() y no depende de
# Streamlit ni de st.secrets.

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Nombre exacto del archivo de la temporada en curso para cada liga dentro
# de data/. Sobreescribir siempre este mismo archivo (en vez de crear uno
# nuevo) evita que aparezcan partidos duplicados.
CURRENT_SEASON_FILE = {
    "SP1": "1789294363276_SP1.csv",
    "E0": "E0.csv",
    "I1": "I1.csv",
    "F1": "F1.csv",
    "D1": "D1.csv",
    "N1": "N1.csv",
    "P1": "P1.csv",
}


def current_season_code(today: date = None) -> str:
    """football-data.co.uk usa códigos de temporada tipo '2627' para 2026-27."""
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{str(start_year)[2:]}{str(start_year + 1)[2:]}"


def fetch_league_csv(league_code: str, season_code: str = None) -> str:
    """Descarga el CSV de la temporada en curso de football-data.co.uk."""
    season_code = season_code or current_season_code()
    url = f"{BASE_URL}/{season_code}/{league_code}.csv"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


def update_league(league_code: str):
    """Descarga y sube a GitHub los datos más recientes de una liga.
    Devuelve (ok: bool, mensaje: str)."""
    from github_sync import put_file

    filename = CURRENT_SEASON_FILE.get(league_code)
    if filename is None:
        return False, f"No sé en qué archivo guardar la liga {league_code}."
    try:
        csv_text = fetch_league_csv(league_code)
    except Exception as e:
        return False, f"No se ha podido descargar de football-data.co.uk: {e}"

    if len(csv_text) < 100:
        return False, "La descarga parece vacía o incompleta; no se ha guardado nada."

    return put_file(f"data/{filename}", csv_text, f"Actualizar {filename} automáticamente")
