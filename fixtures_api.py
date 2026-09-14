"""
Calendario de partidos del día usando la API de football-data.org (no
besoccer.es, que bloquea peticiones desde servidores en la nube).

Necesita un token gratuito guardado en Secrets como FOOTBALL_DATA_TOKEN.
Consíguelo gratis en https://www.football-data.org/client/register
"""
import requests
import streamlit as st

API_BASE = "https://api.football-data.org/v4"

# Código de competición en football-data.org para cada una de nuestras ligas
COMPETITION_CODES = {
    "E0": "PL",
    "SP1": "PD",
    "D1": "BL1",
    "I1": "SA",
    "F1": "FL1",
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


def resolve_team_name(raw_name: str, known_teams: list) -> str | None:
    """Intenta encontrar el nombre que usamos en los CSV para un equipo de
    football-data.org."""
    if raw_name in known_teams:
        return raw_name
    if raw_name in TEAM_ALIASES:
        alias = TEAM_ALIASES[raw_name]
        return alias if alias in known_teams else None

    raw_lower = raw_name.lower()
    for team in known_teams:
        team_lower = team.lower()
        if team_lower in raw_lower or raw_lower in team_lower:
            return team
    return None


def fetch_matches_for_date(date_str: str):
    """Devuelve (partidos, error). partidos es una lista de dicts:
    {league_code, home_raw, away_raw, time, status}. error es None si todo
    fue bien, o un mensaje si algo falló."""
    headers = _headers()
    if headers is None:
        return [], (
            "No hay FOOTBALL_DATA_TOKEN configurado. Ve a Streamlit Cloud > "
            "tu app > Settings > Secrets y añade: "
            "FOOTBALL_DATA_TOKEN = \"tu-token\""
        )

    all_matches = []
    for league_code, comp_code in COMPETITION_CODES.items():
        url = f"{API_BASE}/competitions/{comp_code}/matches"
        try:
            resp = requests.get(
                url, headers=headers,
                params={"dateFrom": date_str, "dateTo": date_str},
                timeout=15,
            )
        except Exception as e:
            return all_matches, f"Error de red consultando {league_code}: {e}"

        if resp.status_code == 429:
            return all_matches, (
                "Límite de peticiones de football-data.org alcanzado "
                "(10 por minuto). Espera un momento y vuelve a intentarlo."
            )
        if resp.status_code != 200:
            return all_matches, f"Error {resp.status_code} en {league_code}: {resp.text[:200]}"

        data = resp.json()
        for m in data.get("matches", []):
            utc_date = m.get("utcDate", "")
            time_str = utc_date[11:16] if len(utc_date) >= 16 else "?"
            all_matches.append({
                "league_code": league_code,
                "home_raw": m["homeTeam"]["name"],
                "away_raw": m["awayTeam"]["name"],
                "time": time_str,
                "status": m.get("status", ""),
            })

    return all_matches, None
