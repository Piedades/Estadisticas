"""
Estadísticas reales por partido (promedios directos de los datos), en vez
de los parámetros internos del modelo Poisson. Mucho más fácil de leer:
"1.8 goles por partido" en vez de "+0.32 en escala logarítmica".
"""
import pandas as pd

# Cada métrica: (nombre a mostrar, columna del equipo local, columna del visitante)
METRIC_COLUMNS = {
    "Córners": ("HC", "AC"),
    "Tiros": ("HS", "AS"),
    "Tiros a puerta": ("HST", "AST"),
    "Tarjetas amarillas": ("HY", "AY"),
}


def team_match_averages(df: pd.DataFrame, team: str) -> dict:
    """Devuelve un diccionario con los promedios reales por partido de un
    equipo: goles a favor/en contra, y cada métrica extra a favor/en contra."""
    home = df[df["HomeTeam"] == team]
    away = df[df["AwayTeam"] == team]
    played = len(home) + len(away)

    stats = {"Partidos jugados": played}
    if played == 0:
        return stats

    goals_for = pd.concat([home["FTHG"], away["FTAG"]])
    goals_against = pd.concat([home["FTAG"], away["FTHG"]])
    stats["Goles a favor"] = goals_for.mean()
    stats["Goles en contra"] = goals_against.mean()

    for label, (h_col, a_col) in METRIC_COLUMNS.items():
        if h_col not in df.columns or a_col not in df.columns:
            continue
        for_ = pd.concat([home[h_col], away[a_col]])
        against_ = pd.concat([home[a_col], away[h_col]])
        stats[f"{label} a favor"] = for_.mean()
        stats[f"{label} en contra"] = against_.mean()

    return stats
