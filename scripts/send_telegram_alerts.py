"""
Aviso diario por Telegram de los partidos (y "value bets") de tus equipos
favoritos. Pensado para ejecutarse como GitHub Action programado
(.github/workflows/telegram-alerts.yml), NO desde la app de Streamlit.

No usa github_sync.py ni depende de que haya una app de Streamlit
corriendo: favorites.json se lee directamente del checkout local (es un
archivo normal ya versionado en el repo), y los tokens se leen de
variables de entorno.

IMPORTANTE: hay que configurar estos secrets en GitHub (Settings > Secrets
and variables > Actions) — son un sitio DISTINTO de los Secrets de
Streamlit Cloud, aunque se llamen igual hay que añadirlos en los dos
sitios si quieres que funcionen tanto aquí como en la app:
  - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (obligatorios, ver telegram_bot.py
    para cómo conseguirlos con @BotFather)
  - FOOTBALL_DATA_TOKEN (obligatorio, para saber qué partidos hay hoy)
  - ODDS_API_TOKEN (opcional: sin él se avisa igual de los partidos de tus
    favoritos, pero sin comparar con cuotas reales ni detectar value bets)
"""
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline_and_model import DixonColesModel, load_league_data  # noqa: E402
from fixtures_api import fetch_matches_for_date, resolve_team_name  # noqa: E402
from odds_api import fetch_odds_for_league, match_odds_to_teams, implied_probs_from_odds  # noqa: E402
from telegram_bot import send_message  # noqa: E402

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
FAVORITES_PATH = Path(__file__).resolve().parent.parent / "favorites.json"

# Mismas 5 ligas que cubre The Odds API en la app (ver LEAGUE_NAME_HINTS en
# odds_api.py) — el resto de ligas que carga la app (Países Bajos, Portugal)
# no tienen cuotas reales disponibles, así que aquí solo se comparan value
# bets en estas 5; el resto de favoritos se avisan igual, sin cuota.
LEAGUES_WITH_ODDS = {
    "E0": "Premier League", "SP1": "LaLiga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1",
}

HALF_LIFE_MONTHS = 15  # mismo valor por defecto que el slider de la app
DECAY = math.log(2) / (HALF_LIFE_MONTHS * 30)
EDGE_THRESHOLD = 0.05  # mismo umbral que usa "Partidos del día" en la app


def load_favorite_teams_by_league() -> dict:
    """{league_code: {equipos favoritos}} juntando los favoritos de TODOS
    los usuarios guardados en favorites.json — la app es de uso personal
    (login desactivado, en la práctica un único usuario real), así que no
    merece la pena complicar el aviso separando por usuario."""
    if not FAVORITES_PATH.exists():
        return {}
    data = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
    by_league = {}
    for _user, leagues in data.items():
        for league_code, teams in leagues.items():
            by_league.setdefault(league_code, set()).update(teams)
    return by_league


def main():
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    fd_token = os.environ.get("FOOTBALL_DATA_TOKEN")
    odds_token = os.environ.get("ODDS_API_TOKEN")

    if not tg_token or not tg_chat_id:
        print("Faltan TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, no se puede avisar. Saliendo.")
        return
    if not fd_token:
        print("Falta FOOTBALL_DATA_TOKEN, no se puede consultar el calendario. Saliendo.")
        return

    favorites_by_league = load_favorite_teams_by_league()
    if not favorites_by_league:
        print("No hay ningún equipo favorito guardado todavía, nada que avisar.")
        return

    today_str = dt.date.today().strftime("%Y-%m-%d")
    matches_today, warnings = fetch_matches_for_date(today_str, token=fd_token)
    for w in warnings:
        print("Aviso:", w)

    lines = []
    for m in matches_today:
        league_code = m["league_code"]
        fav_teams = favorites_by_league.get(league_code)
        if not fav_teams:
            continue  # ni Champions/Europa ni ligas sin favoritos guardados

        df_league = load_league_data(DATA_DIR, league_code)
        if df_league.empty:
            continue
        known_teams = sorted(set(df_league["HomeTeam"]) | set(df_league["AwayTeam"]))
        home = resolve_team_name(m["home_raw"], known_teams)
        away = resolve_team_name(m["away_raw"], known_teams)
        if home is None or away is None:
            continue
        if home not in fav_teams and away not in fav_teams:
            continue

        model = DixonColesModel(decay=DECAY)
        model.fit(df_league)
        if home not in model.teams or away not in model.teams:
            continue
        pred = model.predict_match(home, away)

        line = f"⚽ {m['time']} — {home} vs {away} ({league_code})"

        if odds_token and league_code in LEAGUES_WITH_ODDS:
            odds_matches, odds_err = fetch_odds_for_league(league_code, token=odds_token)
            if odds_err:
                print(f"[{league_code}] cuotas: {odds_err}")
            else:
                om = match_odds_to_teams(odds_matches, home, away, known_teams)
                if om is not None:
                    imp_h, imp_d, imp_a = implied_probs_from_odds(
                        om["odds_home"], om["odds_draw"], om["odds_away"]
                    )
                    edges = [("Local", pred["P(H)"] - imp_h)]
                    if om["odds_draw"]:
                        edges.append(("Empate", pred["P(D)"] - imp_d))
                    edges.append(("Visitante", pred["P(A)"] - imp_a))
                    best_label, best_edge = max(edges, key=lambda e: e[1])
                    if best_edge > EDGE_THRESHOLD:
                        line += (
                            f"\n   💎 Value bet: el modelo ve <b>{best_edge:+.1%}</b> más "
                            f"probable {best_label} de lo que implica la cuota."
                        )

        lines.append(line)

    if not lines:
        print("Ningún favorito juega hoy. No se manda mensaje.")
        return

    text = "<b>⚽ Tus favoritos juegan hoy</b>\n\n" + "\n\n".join(lines)
    ok, msg = send_message(tg_token, tg_chat_id, text)
    print(msg)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
