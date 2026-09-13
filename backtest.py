"""
Backtest walk-forward reutilizable desde la app: entrena siempre solo con
temporadas anteriores a la que se evalúa (nunca mira el futuro) y compara
la probabilidad del modelo contra la cuota media del mercado.
"""
import pandas as pd

from data_pipeline_and_model import DixonColesModel


def _implied_probs(row):
    for h_col, d_col, a_col in [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]:
        if h_col in row and pd.notna(row[h_col]):
            ph, pd_, pa = 1 / row[h_col], 1 / row[d_col], 1 / row[a_col]
            total = ph + pd_ + pa
            return ph / total, pd_ / total, pa / total
    return None, None, None


def run_walk_forward(df: pd.DataFrame, decay: float = 0.0015,
                      edge_threshold: float = 0.05, min_train: int = 200) -> pd.DataFrame:
    seasons = sorted(df["Season"].dropna().unique())
    results = []

    for i, test_season in enumerate(seasons):
        train = df[df["Season"].isin(seasons[:i])]
        test = df[df["Season"] == test_season]
        if len(train) < min_train or len(test) == 0:
            continue

        model = DixonColesModel(decay=decay)
        model.fit(train)

        for _, row in test.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            if home not in model.teams or away not in model.teams:
                continue
            pred = model.predict_match(home, away)
            p_h, p_d, p_a = _implied_probs(row)
            if p_h is None:
                continue
            for outcome, p_model, p_mkt, odds_col in [
                ("H", pred["P(H)"], p_h, "AvgH"),
                ("D", pred["P(D)"], p_d, "AvgD"),
                ("A", pred["P(A)"], p_a, "AvgA"),
            ]:
                if odds_col not in row or pd.isna(row[odds_col]):
                    continue
                edge = p_model - p_mkt
                if edge > edge_threshold:
                    won = row["FTR"] == outcome
                    odds = row[odds_col]
                    pnl = (odds - 1) if won else -1
                    results.append({
                        "season": test_season, "date": row["Date"],
                        "home": home, "away": away, "pick": outcome,
                        "model_p": p_model, "market_p": p_mkt, "edge": edge,
                        "odds": odds, "won": won, "pnl": pnl,
                    })
    return pd.DataFrame(results)
