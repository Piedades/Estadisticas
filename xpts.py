"""
Puntos esperados (xPts): compara los puntos reales de cada equipo en la
temporada en curso con los que "deberia" tener segun las probabilidades
del modelo en cada partido ya jugado. Un equipo con pocos puntos reales
pero muchos xPts esta rindiendo por debajo de lo que sugiere el modelo
(y al reves) - util para detectar candidatos a corregir su racha.

Nota: el modelo usado ya ha "visto" estos partidos al entrenarse (no es
walk-forward como el backtest), asi que es una lectura retrospectiva de
la fuerza actual de cada equipo, no una prediccion a ciegas.
"""
import pandas as pd


def expected_points_table(df: pd.DataFrame, season: str, model) -> pd.DataFrame:
    season_df = df[df["Season"] == season]
    teams = sorted(set(season_df["HomeTeam"]) | set(season_df["AwayTeam"]))
    actual_points = {t: 0 for t in teams}
    xpts = {t: 0.0 for t in teams}
    played = {t: 0 for t in teams}

    for _, row in season_df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        if home not in actual_points or away not in actual_points:
            continue
        hg, ag = row["FTHG"], row["FTAG"]
        if hg > ag:
            actual_points[home] += 3
        elif hg < ag:
            actual_points[away] += 3
        else:
            actual_points[home] += 1
            actual_points[away] += 1
        played[home] += 1
        played[away] += 1

        if home not in model.teams or away not in model.teams:
            continue
        pred = model.predict_match(home, away)
        xpts[home] += 3 * pred["P(H)"] + pred["P(D)"]
        xpts[away] += 3 * pred["P(A)"] + pred["P(D)"]

    result = pd.DataFrame({
        "Equipo": teams,
        "Partidos": [played[t] for t in teams],
        "Puntos reales": [actual_points[t] for t in teams],
        "xPts": [round(xpts[t], 1) for t in teams],
    })
    result["Diferencia"] = (result["Puntos reales"] - result["xPts"]).round(1)
    return result.sort_values("Diferencia", ascending=True).reset_index(drop=True)
