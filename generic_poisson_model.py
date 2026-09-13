"""
Modelo genérico ataque/defensa (estilo Dixon-Coles simplificado, sin la
corrección de marcadores bajos que solo aplica a goles) para cualquier
métrica de conteo: córners, tarjetas, tiros, tiros a puerta.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


class GenericPoissonModel:
    def __init__(self, home_col, away_col, decay=0.0015, label="metric"):
        self.home_col = home_col
        self.away_col = away_col
        self.decay = decay
        self.label = label
        self.teams = None
        self.params = None

    def _time_weights(self, dates):
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.values
        return np.exp(-self.decay * days_ago)

    def fit(self, df: pd.DataFrame):
        df = df.dropna(subset=[self.home_col, self.away_col]).copy()
        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}
        weights = self._time_weights(df["Date"])

        home_idx = df["HomeTeam"].map(idx).values
        away_idx = df["AwayTeam"].map(idx).values
        h_val = df[self.home_col].values
        a_val = df[self.away_col].values

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.1]])

        def neg_log_likelihood(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defense[home_idx])
            ll = poisson.logpmf(h_val, lam) + poisson.logpmf(a_val, mu)
            return -np.sum(ll * weights)

        constraints = {"type": "eq", "fun": lambda p: np.mean(p[:n])}
        result = minimize(neg_log_likelihood, x0, constraints=constraints, method="SLSQP",
                           options={"maxiter": 200, "ftol": 1e-6})
        self.params = result.x
        return self

    def team_ratings(self):
        n = len(self.teams)
        attack = self.params[:n]
        defense = self.params[n:2 * n]
        return pd.DataFrame({
            "team": self.teams,
            f"{self.label}_attack": attack,
            f"{self.label}_defense": defense,
        }).sort_values(f"{self.label}_attack", ascending=False)

    def predict_match(self, home_team, away_team):
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}
        attack = self.params[:n]
        defense = self.params[n:2 * n]
        home_adv = self.params[2 * n]
        i, j = idx[home_team], idx[away_team]
        lam = np.exp(attack[i] + defense[j] + home_adv)
        mu = np.exp(attack[j] + defense[i])
        return {"expected_home": lam, "expected_away": mu, "expected_total": lam + mu}
