"""
Equipos favoritos por usuario: guarda, para cada usuario y liga, la lista
de equipos marcados como favoritos en favorites.json dentro del repositorio
(mismo mecanismo que bets_log.csv y users.json, via github_sync).
"""
import json

from github_sync import get_file, put_file

FAVORITES_PATH = "favorites.json"


def load_favorites() -> dict:
    content, _ = get_file(FAVORITES_PATH)
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def get_user_favorites(username: str, league_code: str) -> list:
    data = load_favorites()
    return data.get(username, {}).get(league_code, [])


def save_user_favorites(username: str, league_code: str, teams: list):
    data = load_favorites()
    data.setdefault(username, {})[league_code] = teams
    ok, msg = put_file(
        FAVORITES_PATH, json.dumps(data, ensure_ascii=False, indent=2),
        f"Actualizar favoritos de {username} en {league_code}",
    )
    return ok, msg
