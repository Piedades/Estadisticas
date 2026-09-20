"""
Cuotas de mercado reales (The Odds API) para comparar con la probabilidad
que da nuestro modelo y detectar "value bets" — apuestas donde el modelo ve
más probabilidad de un resultado que la que implica la cuota del mercado.

Necesita un token gratuito guardado en Secrets como ODDS_API_TOKEN.
Consíguelo en https://the-odds-api.com/ (plan gratuito, con límite mensual
de peticiones — por eso esto se consulta solo cuando el usuario lo pide a
propósito, no automáticamente al buscar partidos).

Solo cubrimos las 5 grandes ligas (no Champions/Europa/Países Bajos/
Portugal): con el límite mensual del plan gratuito, ampliarlo a más
competiciones agotaría la cuota enseguida para poco beneficio extra.
"""
import streamlit as st
import requests

from fixtures_api import resolve_team_name

API_BASE = "https://api.the-odds-api.com/v4"

# A qué debe "sonar" el title/description de una liga en el catálogo de
# The Odds API (GET /v4/sports) para identificarla como nuestra. Se resuelve
# en vivo (no hardcodeamos las "sport keys" tipo soccer_spain_la_liga:
# preferimos buscarlas por nombre a arriesgarnos a que el identificador
# exacto haya cambiado y la liga deje de encontrarse en silencio).
LEAGUE_NAME_HINTS = {
    "E0": ["premier league"],
    "SP1": ["la liga", "laliga", "primera división", "primera division"],
    "D1": ["bundesliga"],
    "I1": ["serie a"],
    "F1": ["ligue 1"],
}


def _token():
    return st.secrets.get("ODDS_API_TOKEN")


@st.cache_data(show_spinner=False, ttl="6h")
def _discover_sport_keys() -> dict:
    """GET /v4/sports una vez (se cachea) y empareja cada una de nuestras 5
    ligas con su "sport key" real buscando por nombre. Devuelve
    {league_code: sport_key} — una liga que no se encuentre simplemente no
    sale en el dict, no rompe nada."""
    token = _token()
    if not token:
        return {}
    try:
        resp = requests.get(f"{API_BASE}/sports", params={"apiKey": token}, timeout=15)
    except Exception:
        return {}
    if resp.status_code != 200:
        return {}
    sports = resp.json()

    found = {}
    for league_code, hints in LEAGUE_NAME_HINTS.items():
        for sport in sports:
            title = (sport.get("title") or "").lower()
            desc = (sport.get("description") or "").lower()
            group = (sport.get("group") or "").lower()
            if group != "soccer":
                continue
            if any(h in title or h in desc for h in hints):
                found[league_code] = sport["key"]
                break
    return found


def fetch_odds_for_league(league_code: str):
    """Devuelve (partidos_con_cuota, error). partidos_con_cuota es una lista
    de dicts {home_raw, away_raw, odds_home, odds_draw, odds_away} — la
    cuota media 1X2 entre las casas que devuelve la API para esa región.
    error es None si fue bien, o un mensaje explicando qué falló."""
    token = _token()
    if not token:
        return [], (
            "No hay ODDS_API_TOKEN configurado. Ve a Streamlit Cloud > tu "
            "app > Settings > Secrets y añade: ODDS_API_TOKEN = \"tu-token\" "
            "(consíguelo gratis en the-odds-api.com)."
        )

    sport_keys = _discover_sport_keys()
    sport_key = sport_keys.get(league_code)
    if sport_key is None:
        return [], (
            "No he encontrado esta liga en el catálogo de The Odds API "
            "(puede que tu token no dé acceso a fútbol, o que hayan "
            "renombrado la competición)."
        )

    try:
        resp = requests.get(
            f"{API_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": token,
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )
    except Exception as e:
        return [], f"Error de red consultando cuotas: {e}"

    if resp.status_code == 401:
        return [], "Token de The Odds API inválido o caducado."
    if resp.status_code == 429:
        return [], "Límite mensual de peticiones de The Odds API agotado."
    if resp.status_code != 200:
        return [], f"Error {resp.status_code} consultando cuotas: {resp.text[:200]}"

    matches = []
    for event in resp.json():
        home_raw = event.get("home_team")
        away_raw = event.get("away_team")
        if not home_raw or not away_raw:
            continue
        # Media de la cuota 1X2 entre todas las casas que trae la respuesta,
        # para no depender de una sola casa de apuestas.
        h_odds, d_odds, a_odds = [], [], []
        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name, price = outcome.get("name"), outcome.get("price")
                    if name == home_raw:
                        h_odds.append(price)
                    elif name == away_raw:
                        a_odds.append(price)
                    elif name == "Draw":
                        d_odds.append(price)
        if not h_odds or not a_odds:
            continue
        matches.append({
            "home_raw": home_raw,
            "away_raw": away_raw,
            "odds_home": sum(h_odds) / len(h_odds),
            "odds_draw": (sum(d_odds) / len(d_odds)) if d_odds else None,
            "odds_away": sum(a_odds) / len(a_odds),
            "n_bookmakers": len(event.get("bookmakers", [])),
        })
    return matches, None


def match_odds_to_teams(odds_matches: list, home: str, away: str, known_teams: list):
    """Busca, dentro de lo que devolvió fetch_odds_for_league, la cuota del
    partido home-away (ya resueltos a nuestros nombres CSV). Reutiliza
    resolve_team_name para que "Real Madrid" (The Odds API) empareje con
    "Real Madrid" (nuestro CSV) igual que ya hacemos con football-data.org."""
    for om in odds_matches:
        oh = resolve_team_name(om["home_raw"], known_teams)
        oa = resolve_team_name(om["away_raw"], known_teams)
        if oh == home and oa == away:
            return om
    return None


def implied_probs_from_odds(odds_home: float, odds_draw, odds_away: float):
    """Cuotas -> probabilidades implícitas, quitando el margen de la casa
    (overround) normalizando para que sumen 100%."""
    ph = 1 / odds_home
    pa = 1 / odds_away
    pd_ = 1 / odds_draw if odds_draw else 0.0
    total = ph + pd_ + pa
    if pd_:
        return ph / total, pd_ / total, pa / total
    return ph / total, None, pa / total
