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


def team_line_hit_rate(df: pd.DataFrame, team: str, h_col: str, a_col: str, line: float, venue: str = "Ambos") -> dict:
    """Estadística de "cuántas veces se ha superado esta línea": para los
    mercados de córners/tiros/tiros a puerta/tarjetas/goles la cuota es
    sobre el TOTAL del partido (local + visitante sumados, no solo lo que
    hace el equipo elegido — confirmado por GenericPoissonModel.predict_match,
    que da expected_total = lam + mu), así que esto mira, de los partidos en
    los que jugó `team`, en cuántos el total del partido superó `line`.

    venue: "Ambos" (todos los partidos del equipo), "Local" (solo cuando
    jugó en casa) o "Visitante" (solo cuando jugó fuera).

    Devuelve {"partidos": 0} si no hay datos suficientes, o
    {"partidos", "over_pct", "under_pct", "media"} si los hay."""
    if venue == "Local":
        sub = df[df["HomeTeam"] == team]
    elif venue == "Visitante":
        sub = df[df["AwayTeam"] == team]
    else:
        sub = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]

    if h_col not in sub.columns or a_col not in sub.columns:
        return {"partidos": 0}
    sub = sub.dropna(subset=[h_col, a_col])
    if sub.empty:
        return {"partidos": 0}

    totals = sub[h_col] + sub[a_col]
    n = len(totals)
    over = int((totals > line).sum())
    return {
        "partidos": n,
        "over_pct": over / n,
        "under_pct": 1 - (over / n),
        "media": totals.mean(),
    }
