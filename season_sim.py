"""
Simulador de temporada completa: simula todos los partidos que quedan de
la temporada en curso, miles de veces seguidas (Monte Carlo), para sacar
probabilidades de título, puestos europeos y descenso.

Los datos históricos (football-data.co.uk) no incluyen el calendario de
partidos futuros, así que reconstruimos los que faltan asumiendo una liga
de doble vuelta estándar (cada equipo juega dos veces contra cada rival,
una en casa y otra fuera) y restando los que ya se han jugado esta
temporada. Es una aproximación razonable, no el calendario real exacto.
"""
import numpy as np
import pandas as pd

from data_pipeline_and_model import DixonColesModel


def get_remaining_fixtures(df: pd.DataFrame, season: str):
    season_df = df[df["Season"] == season]
    teams = sorted(set(season_df["HomeTeam"]) | set(season_df["AwayTeam"]))
    played = set(zip(season_df["HomeTeam"], season_df["AwayTeam"]))
    all_fixtures = [(h, a) for h in teams for a in teams if h != a]
    remaining = [f for f in all_fixtures if f not in played]
    return remaining, teams


def current_standings(df: pd.DataFrame, season: str, teams: list) -> pd.DataFrame:
    season_df = df[df["Season"] == season]
    points = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    played = {t: 0 for t in teams}
    for _, row in season_df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        if h not in points or a not in points:
            continue
        hg, ag = row["FTHG"], row["FTAG"]
        gf[h] += hg
        ga[h] += ag
        played[h] += 1
        gf[a] += ag
        ga[a] += hg
        played[a] += 1
        if hg > ag:
            points[h] += 3
        elif hg < ag:
            points[a] += 3
        else:
            points[h] += 1
            points[a] += 1
    return pd.DataFrame({
        "team": teams,
        "played": [played[t] for t in teams],
        "points": [points[t] for t in teams],
        "gf": [gf[t] for t in teams],
        "ga": [ga[t] for t in teams],
    })


def simulate_season(df: pd.DataFrame, season: str, decay: float,
                     n_sims: int = 2000, top_europe: int = 4, relegation: int = 3,
                     seed: int = 42) -> pd.DataFrame:
    remaining, teams = get_remaining_fixtures(df, season)
    base = current_standings(df, season, teams).set_index("team").reindex(teams)

    model = DixonColesModel(decay=decay)
    model.fit(df)

    fixture_lambdas = []
    for home, away in remaining:
        if home not in model.teams or away not in model.teams:
            continue
        pred = model.predict_match(home, away)
        fixture_lambdas.append((home, away, pred["lambda_home"], pred["lambda_away"]))

    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)

    base_points = base["points"].values.astype(float)
    base_gd = (base["gf"] - base["ga"]).values.astype(float)

    points = np.tile(base_points, (n_sims, 1))
    gd = np.tile(base_gd, (n_sims, 1))

    for home, away, lh, la in fixture_lambdas:
        hg = rng.poisson(lh, size=n_sims)
        ag = rng.poisson(la, size=n_sims)
        hi, ai = idx[home], idx[away]
        gd[:, hi] += hg - ag
        gd[:, ai] += ag - hg
        home_win = hg > ag
        away_win = hg < ag
        draw = hg == ag
        points[home_win, hi] += 3
        points[away_win, ai] += 3
        points[draw, hi] += 1
        points[draw, ai] += 1

    champion_count = np.zeros(n, dtype=int)
    europe_count = np.zeros(n, dtype=int)
    relegation_count = np.zeros(n, dtype=int)

    for s in range(n_sims):
        order = np.lexsort((-gd[s], -points[s]))
        champion_count[order[0]] += 1
        europe_count[order[:top_europe]] += 1
        relegation_count[order[-relegation:]] += 1

    result = pd.DataFrame({
        "team": teams,
        "Puntos actuales": base_points.astype(int),
        "P(Campeón)": champion_count / n_sims,
        "P(Puestos europeos)": europe_count / n_sims,
        "P(Descenso)": relegation_count / n_sims,
    }).sort_values("P(Campeón)", ascending=False).reset_index(drop=True)

    return result
