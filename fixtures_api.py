"""
Calendario de partidos del día usando la API de football-data.org (no
besoccer.es, que bloquea peticiones desde servidores en la nube).

Necesita un token gratuito guardado en Secrets como FOOTBALL_DATA_TOKEN.
Consíguelo gratis en https://www.football-data.org/client/register
"""
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import streamlit as st

LOCAL_TZ = ZoneInfo("Europe/Madrid")

API_BASE = "https://api.football-data.org/v4"

# Código de competición en football-data.org para cada una de nuestras ligas
COMPETITION_CODES = {
    "E0": "PL",
    "SP1": "PD",
    "D1": "BL1",
    "I1": "SA",
    "F1": "FL1",
}

# Competiciones europeas: no son "nuestras ligas" (los dos equipos pueden ser
# de ligas domésticas distintas), así que van aparte con sus propios códigos
# especiales ("UCL"/"UEL") en vez de un código de liga (SP1, E0...). El código
# de football-data.org para la Champions League es "CL"; el de la Europa
# League es "EL" — ambas en el Free Tier según su página de cobertura, pero
# football-data.org podría cambiar esto, así que fetch_matches_for_date no
# aborta el resto de competiciones si una de las dos falla.
EUROPEAN_COMPETITION_CODES = {
    "UCL": "CL",
    "UEL": "EL",
}
EUROPEAN_DISPLAY_NAMES = {
    "UCL": "🏆 Champions League",
    "UEL": "🥈 Europa League",
}

# football-data.org usa nombres oficiales largos ("Real Madrid CF") que no
# siempre coinciden con los nombres cortos de nuestros CSV ("Real Madrid").
# Casos conocidos que no se resuelven solo con "contiene la palabra":
TEAM_ALIASES = {
    "Atlético de Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "RC Deportivo La Coruña": "La Coruna",
    "Deportivo Alavés": "Alaves",
    "CA Osasuna": "Osasuna",
    "Rayo Vallecano de Madrid": "Vallecano",
    "Real Sociedad de Fútbol": "Sociedad",
    "Bayern München": "Bayern Munich",
    "Borussia Mönchengladbach": "M'gladbach",
    "1. FC Köln": "FC Koln",
    "1. FC Union Berlin": "Union Berlin",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "SV Werder Bremen": "Werder Bremen",
    "1. FSV Mainz 05": "Mainz",
    "FC St. Pauli 1910": "St Pauli",
    "Hamburger SV": "Hamburg",
    "SpVgg Greuther Fürth": "Greuther Furth",
    "Paris Saint-Germain FC": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
    "LOSC Lille": "Lille",
    "Stade Rennais FC 1901": "Rennes",
    "RC Strasbourg Alsace": "Strasbourg",
    "Racing Club de Lens": "Lens",
    "Stade Brestois 29": "Brest",
    "Toulouse FC": "Toulouse",
    "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "AC Monza": "Monza",
    "Hellas Verona FC": "Verona",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Atalanta BC": "Atalanta",
    "US Sassuolo Calcio": "Sassuolo",
    "Bologna FC 1909": "Bologna",
    "ACF Fiorentina": "Fiorentina",
    "Udinese Calcio": "Udinese",
    "Cagliari Calcio": "Cagliari",
    "US Lecce": "Lecce",
    "Manchester United FC": "Man United",
    "Manchester City FC": "Man City",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "Brighton & Hove Albion FC": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Crystal Palace FC": "Crystal Palace",
    "Leeds United FC": "Leeds",
}


def _headers():
    token = st.secrets.get("FOOTBALL_DATA_TOKEN")
    if not token:
        return None
    return {"X-Auth-Token": token}


def _normalize(s: str) -> str:
    """Minúsculas y sin tildes/diacríticos, para comparar nombres de equipo
    sin que un acento (o que la API y nuestro texto usen formas Unicode
    distintas para la misma tilde) haga fallar el emparejamiento — por
    ejemplo "Málaga" vs "Malaga" o "Coruña" vs "Coruna"."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip()


_ALIASES_NORM = {_normalize(k): v for k, v in TEAM_ALIASES.items()}


def resolve_team_name(raw_name: str, known_teams: list) -> str | None:
    """Intenta encontrar el nombre que usamos en los CSV para un equipo de
    football-data.org."""
    if raw_name in known_teams:
        return raw_name

    raw_norm = _normalize(raw_name)

    # Alias exacto (comparando ya sin tildes, para no depender de que el
    # nombre oficial de la API tenga exactamente los mismos caracteres que
    # escribimos a mano en TEAM_ALIASES).
    alias = _ALIASES_NORM.get(raw_norm)
    if alias is not None:
        return alias if alias in known_teams else None

    # Alias por subcadena: por si el nombre oficial trae alguna palabra de
    # más ("Club Atlético de Madrid" en vez de "Atlético de Madrid").
    for alias_key_norm, alias_value in _ALIASES_NORM.items():
        if alias_key_norm in raw_norm or raw_norm in alias_key_norm:
            return alias_value if alias_value in known_teams else None

    # Último recurso: coincidencia directa por subcadena con el nombre corto
    # que usa football-data.co.uk en nuestros CSV.
    for team in known_teams:
        team_norm = _normalize(team)
        if team_norm in raw_norm or raw_norm in team_norm:
            return team
    return None


def fetch_matches_for_date(date_str: str):
    """Devuelve (partidos, avisos). partidos es una lista de dicts:
    {league_code, home_raw, away_raw, time, status}. avisos es una lista de
    mensajes (vacía si todo fue bien) — un fallo en UNA competición (p. ej.
    si algún día football-data.org cambiara el código de la Europa League)
    ya no cancela las demás, solo se avisa y se sigue con el resto."""
    headers = _headers()
    if headers is None:
        return [], [
            "No hay FOOTBALL_DATA_TOKEN configurado. Ve a Streamlit Cloud > "
            "tu app > Settings > Secrets y añade: "
            "FOOTBALL_DATA_TOKEN = \"tu-token\""
        ]

    all_competitions = {**COMPETITION_CODES, **EUROPEAN_COMPETITION_CODES}
    all_matches = []
    warnings = []
    for league_code, comp_code in all_competitions.items():
        url = f"{API_BASE}/competitions/{comp_code}/matches"
        try:
            resp = requests.get(
                url, headers=headers,
                params={"dateFrom": date_str, "dateTo": date_str},
                timeout=15,
            )
        except Exception as e:
            warnings.append(f"Error de red consultando {league_code}: {e}")
            continue

        if resp.status_code == 429:
            warnings.append(
                "Límite de peticiones de football-data.org alcanzado "
                "(10 por minuto); puede que falten partidos de alguna "
                "competición. Espera un momento y vuelve a intentarlo."
            )
            break  # el resto de peticiones también darían 429, no seguimos
        if resp.status_code != 200:
            warnings.append(f"No se pudo consultar {league_code} (error {resp.status_code}).")
            continue

        data = resp.json()
        for m in data.get("matches", []):
            utc_date = m.get("utcDate", "")
            time_str = "?"
            if utc_date:
                try:
                    # football-data.org da la hora en UTC ("...Z"); hay que
                    # convertirla a hora de España (CET/CEST según la época
                    # del año) para que coincida con la hora real del partido.
                    dt_utc = datetime.strptime(utc_date, "%Y-%m-%dT%H:%M:%SZ")
                    dt_utc = dt_utc.replace(tzinfo=ZoneInfo("UTC"))
                    time_str = dt_utc.astimezone(LOCAL_TZ).strftime("%H:%M")
                except ValueError:
                    time_str = utc_date[11:16] if len(utc_date) >= 16 else "?"
            all_matches.append({
                "league_code": league_code,
                "home_raw": m["homeTeam"]["name"],
                "away_raw": m["awayTeam"]["name"],
                "time": time_str,
                "status": m.get("status", ""),
            })

    return all_matches, warnings
