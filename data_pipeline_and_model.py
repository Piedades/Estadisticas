"""
Pipeline de datos + modelo predictivo para las Big 5 ligas europeas.
Fuente: football-data.co.uk (formato SP1/SP2/E0/I1/D1/F1)

Uso:
    python data_pipeline_and_model.py --data_dir ./data --league SP1

Requisitos: pandas, numpy, scipy
    pip install pandas numpy scipy --break-system-packages
"""

import glob
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ----------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA
# ----------------------------------------------------------------------

def season_from_date(d: pd.Timestamp) -> str:
    """Asigna temporada tipo '2024-25' según la fecha (temporada = ago-jun)."""
    if pd.isna(d):
        return None
    return f"{d.year}-{str(d.year + 1)[2:]}" if d.month >= 7 else f"{d.year - 1}-{str(d.year)[2:]}"


def load_league_data(data_dir: str, league_code: str = None) -> pd.DataFrame:
    """Carga y concatena todos los CSV de un directorio (opcionalmente filtrando por liga)."""
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        if "Div" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df["Season"] = df["Date"].apply(season_from_date)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    if league_code:
        all_df = all_df[all_df["Div"] == league_code]
    all_df = all_df.dropna(subset=["FTHG", "FTAG", "Date"]).sort_values("Date")
    return all_df.reset_index(drop=True)


def implied_probabilities(row, h_col="AvgH", d_col="AvgD", a_col="AvgA"):
    """Convierte cuotas medias en probabilidades implícitas normalizadas (sin overround)."""
    ph, pd_, pa = 1 / row[h_col], 1 / row[d_col], 1 / row[a_col]
    total = ph + pd_ + pa
    return ph / total, pd_ / total, pa / total


# ----------------------------------------------------------------------
# 2. MODELO POISSON CON AJUSTE DIXON-COLES
# ----------------------------------------------------------------------
# Cada equipo tiene un parámetro de ataque y uno de defensa.
# lambda_home = exp(attack_home + defense_away + home_advantage)
# lambda_away = exp(attack_away + defense_home)
# Ajuste de Dixon-Coles (rho) corrige la dependencia entre marcadores bajos (0-0,1-0,0-1,1-1).

class DixonColesModel:
    def __init__(self, decay: float = 0.0018):
        """decay: tasa de decaimiento temporal (peso menor a partidos antiguos)."""
        self.decay = decay
        self.teams = None
        self.params = None

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.values
        return np.exp(-self.decay * days_ago)

    @staticmethod
    def _tau(x, y, lambda_x, mu_y, rho):
        """Función de corrección de Dixon-Coles para marcadores 0-0, 0-1, 1-0, 1-1."""
        if x == 0 and y == 0:
            return 1 - lambda_x * mu_y * rho
        elif x == 0 and y == 1:
            return 1 + lambda_x * rho
        elif x == 1 and y == 0:
            return 1 + mu_y * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def fit(self, df: pd.DataFrame):
        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        self.teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}
        weights = self._time_weights(df["Date"])

        home_idx = df["HomeTeam"].map(idx).values
        away_idx = df["AwayTeam"].map(idx).values
        hg = df["FTHG"].values
        ag = df["FTAG"].values

        # x0: [attack_1..n, defense_1..n, home_adv, rho]
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [-0.05]])

        def neg_log_likelihood(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defense[home_idx])

            ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
            tau = np.array([
                self._tau(x, y, lx, my, rho)
                for x, y, lx, my in zip(hg, ag, lam, mu)
            ])
            tau = np.clip(tau, 1e-10, None)
            ll += np.log(tau)
            return -np.sum(ll * weights)

        # restricción: media de ataque = 0 (para identificabilidad)
        constraints = {"type": "eq", "fun": lambda p: np.mean(p[:n])}
        result = minimize(neg_log_likelihood, x0, constraints=constraints, method="SLSQP",
                           options={"maxiter": 200, "ftol": 1e-6})
        self.params = result.x
        return self

    def predict_match(self, home_team: str, away_team: str, max_goals: int = 8):
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}
        attack = self.params[:n]
        defense = self.params[n:2 * n]
        home_adv = self.params[2 * n]
        rho = self.params[2 * n + 1]

        i, j = idx[home_team], idx[away_team]
        lam = np.exp(attack[i] + defense[j] + home_adv)
        mu = np.exp(attack[j] + defense[i])

        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                p = poisson.pmf(x, lam) * poisson.pmf(y, mu) * self._tau(x, y, lam, mu, rho)
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


# ----------------------------------------------------------------------
# 3. EJEMPLO DE USO
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--league", default="SP1")
    parser.add_argument("--home", default=None)
    parser.add_argument("--away", default=None)
    args = parser.parse_args()

    df = load_league_data(args.data_dir, args.league)
    print(f"Cargados {len(df)} partidos de {args.league} "
          f"({df['Date'].min().date()} a {df['Date'].max().date()})")

    model = DixonColesModel(decay=0.0018)
    model.fit(df)

    home = args.home or df["HomeTeam"].iloc[-1]
    away = args.away or df["AwayTeam"].iloc[-1]
    pred = model.predict_match(home, away)

    print(f"\n{pred['home_team']} vs {pred['away_team']}")
    print(f"  Goles esperados: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")
    print(f"  P(Local): {pred['P(H)']:.1%}  P(Empate): {pred['P(D)']:.1%}  P(Visitante): {pred['P(A)']:.1%}")
