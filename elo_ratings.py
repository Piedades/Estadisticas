"""
Rating Elo dinámico para fútbol, actualizado partido a partido.
Incluye ventaja de local y margen de victoria (opcional).
"""
import pandas as pd
import numpy as np


class EloTracker:
    def __init__(self, k=20, home_adv=60, initial=1500):
        self.k = k
        self.home_adv = home_adv
        self.initial = initial
        self.ratings = {}
        self.history = []

    def _get(self, team):
        return self.ratings.setdefault(team, self.initial)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_match(self, date, home, away, result):
        """result: 'H', 'D', 'A' desde perspectiva del marcador."""
        r_home = self._get(home)
        r_away = self._get(away)
        exp_home = self.expected_score(r_home + self.home_adv, r_away)

        actual_home = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)

        delta = self.k * (actual_home - exp_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta

        self.history.append({
            "date": date, "home": home, "away": away,
            "elo_home_pre": r_home, "elo_away_pre": r_away,
            "exp_home_win": exp_home,
        })

    def fit(self, df: pd.DataFrame):
        for _, row in df.sort_values("Date").iterrows():
            self.update_match(row["Date"], row["HomeTeam"], row["AwayTeam"], row["FTR"])
        return self

    def current_table(self):
        return pd.DataFrame(
            sorted(self.ratings.items(), key=lambda x: -x[1]),
            columns=["team", "elo"]
        )

    def win_prob(self, home, away):
        r_home = self._get(home)
        r_away = self._get(away)
        return self.expected_score(r_home + self.home_adv, r_away)
