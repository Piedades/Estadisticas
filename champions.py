"""
Predicción de partidos entre equipos de ligas distintas (por ejemplo, cruces de
la fase de liga de la Champions). No se entrena un modelo aparte con datos de la
propia Champions -la app no tiene ese histórico-, sino que se reutilizan los
parámetros de ataque/defensa que cada equipo ya tiene en su propia liga doméstica.

Aviso importante (que también se muestra en la app): como cada modelo se ajusta
con los goles/córners/tarjetas reales de su propia liga, la diferencia de nivel
general entre ligas (p. ej. más goles de media en una que en otra) ya queda
incorporada en la escala de cada modelo. Lo que NO se corrige es la distancia de
calidad "top a top" entre ligas: un equipo puntero de una liga más floja puede
salir mejor parado de lo que sería en la realidad frente a un rival de una liga
más competitiva. Son predicciones a tratar con más cautela que un partido dentro
de la misma liga.
"""
import numpy as np
from scipy.stats import poisson

from data_pipeline_and_model import DixonColesModel


def predict_cross_goals(model_home: DixonColesModel, home_team: str,
                         model_away: DixonColesModel, away_team: str,
                         max_goals: int = 8) -> dict:
    """Igual que DixonColesModel.predict_match, pero el equipo local usa su propio
    modelo de liga (ataque + defensa + ventaja de local) y el visitante el suyo."""
    n_home = len(model_home.teams)
    n_away = len(model_away.teams)
    i = model_home.teams.index(home_team)
    j = model_away.teams.index(away_team)

    attack_home = model_home.params[i]
    defense_home = model_home.params[n_home + i]
    home_adv = model_home.params[2 * n_home]
    rho_home = model_home.params[2 * n_home + 1]

    attack_away = model_away.params[j]
    defense_away = model_away.params[n_away + j]
    rho_away = model_away.params[2 * n_away + 1]

    # Corrección Dixon-Coles: promedio de la de ambas ligas, no hay una "correcta".
    rho = (rho_home + rho_away) / 2

    lam = np.exp(attack_home + defense_away + home_adv)
    mu = np.exp(attack_away + defense_home)

    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = poisson.pmf(x, lam) * poisson.pmf(y, mu) * DixonColesModel._tau(x, y, lam, mu, rho)
            score_matrix[x, y] = p
    score_matrix /= score_matrix.sum()

    p_home = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_away = np.triu(score_matrix, 1).sum()

    return {
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": lam,
        "lambda_away": mu,
        "P(H)": p_home,
        "P(D)": p_draw,
        "P(A)": p_away,
        "score_matrix": score_matrix,
    }


def predict_cross_generic(model_home, home_team: str, model_away, away_team: str) -> dict:
    """Misma idea que predict_cross_goals, pero para los modelos genéricos de
    córners/tiros/tiros a puerta/tarjetas (GenericPoissonModel), que no tienen
    corrección de Dixon-Coles."""
    n_home = len(model_home.teams)
    n_away = len(model_away.teams)
    i = model_home.teams.index(home_team)
    j = model_away.teams.index(away_team)

    attack_home = model_home.params[i]
    defense_home = model_home.params[n_home + i]
    home_adv = model_home.params[2 * n_home]

    attack_away = model_away.params[j]
    defense_away = model_away.params[n_away + j]

    lam = np.exp(attack_home + defense_away + home_adv)
    mu = np.exp(attack_away + defense_home)
    return {"expected_home": lam, "expected_away": mu, "expected_total": lam + mu}
