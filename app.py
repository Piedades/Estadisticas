"""
Dashboard interactivo — Modelo predictivo Big 5 Ligas Europeas
Ejecutar con: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from data_pipeline_and_model import DixonColesModel, load_league_data
from generic_poisson_model import GenericPoissonModel
from elo_ratings import EloTracker
from backtest import run_walk_forward

st.set_page_config(page_title="Big 5 Ligas — Panel de predicción", layout="wide")

# ---------------------------------------------------------------
# Acceso protegido por contraseña
# ---------------------------------------------------------------
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "futbol2026")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Big 5 Ligas — Acceso")
    pwd = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        if pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.caption(
        "Contraseña por defecto: futbol2026. Cámbiala en Streamlit Cloud > "
        "Settings > Secrets añadiendo: APP_PASSWORD = \"tu-contraseña\""
    )
    st.stop()

LEAGUES = {
    "SP1": "España — LaLiga",
    "E0": "Inglaterra — Premier League",
    "I1": "Italia — Serie A",
    "F1": "Francia — Ligue 1",
    "D1": "Alemania — Bundesliga",
}

DATA_DIR = "./data"

METRICS = {
    "Córners": ("HC", "AC"),
    "Tiros": ("HS", "AS"),
    "Tiros a puerta": ("HST", "AST"),
    "Tarjetas amarillas": ("HY", "AY"),
}


@st.cache_data(show_spinner="Cargando datos de la liga...")
def get_league_df(league_code: str) -> pd.DataFrame:
    return load_league_data(DATA_DIR, league_code)


@st.cache_resource(show_spinner="Entrenando modelo de goles...")
def get_goals_model(league_code: str, decay: float) -> DixonColesModel:
    df = get_league_df(league_code)
    model = DixonColesModel(decay=decay)
    model.fit(df)
    return model


@st.cache_resource(show_spinner="Entrenando modelos de córners/tiros/tarjetas...")
def get_metric_models(league_code: str):
    df = get_league_df(league_code)
    models = {}
    for label, (h_col, a_col) in METRICS.items():
        if h_col not in df.columns:
            continue
        m = GenericPoissonModel(h_col, a_col, label=label)
        m.fit(df)
        models[label] = m
    return models


@st.cache_resource(show_spinner="Calculando ratings Elo...")
def get_elo(league_code: str) -> EloTracker:
    df = get_league_df(league_code)
    elo = EloTracker(k=20, home_adv=60)
    elo.fit(df)
    return elo


@st.cache_data(show_spinner="Calculando backtest walk-forward (puede tardar un poco)...")
def get_backtest(league_code: str, decay: float) -> pd.DataFrame:
    df = get_league_df(league_code)
    return run_walk_forward(df, decay=decay)


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("⚽ Big 5 Ligas")
league_label = st.sidebar.selectbox("Liga", list(LEAGUES.values()))
league_code = [k for k, v in LEAGUES.items() if v == league_label][0]

decay = st.sidebar.slider(
    "Peso de partidos recientes (decay)", 0.0005, 0.005, 0.0015, 0.0005,
    help="Más alto = el modelo da más peso a los partidos recientes y olvida antes los viejos."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Modelo Poisson / Dixon-Coles entrenado con tus datos locales de football-data.co.uk. "
    "Nada de esto sale de tu ordenador."
)

if st.sidebar.button("Cerrar sesión"):
    st.session_state["authenticated"] = False
    st.rerun()

df = get_league_df(league_code)
goals_model = get_goals_model(league_code, decay)
metric_models = get_metric_models(league_code)
elo = get_elo(league_code)

st.title(f"{league_label}")
st.caption(f"{len(df)} partidos cargados · {df['Date'].min().date()} a {df['Date'].max().date()}")

tab_pred, tab_team, tab_elo, tab_ratings, tab_stats, tab_backtest = st.tabs(
    ["Predecir partido", "Ficha de equipo", "Ranking Elo",
     "Ratings por equipo", "Estadísticas de la liga", "Rendimiento histórico"]
)

# ---------------------------------------------------------------
# TAB 1: Predicción de partido (+ comparador de cuotas + exportar)
# ---------------------------------------------------------------
with tab_pred:
    teams = sorted(goals_model.teams)
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipo local", teams, index=0, key="home_team")
    with col2:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("Equipo visitante", away_options, index=0, key="away_team")

    if st.button("Calcular predicción", type="primary"):
        st.session_state["show_prediction"] = True

    if st.session_state.get("show_prediction"):
        pred = goals_model.predict_match(home, away)

        st.subheader(f"{home} vs {away}")

        c1, c2, c3 = st.columns(3)
        c1.metric("P(Local)", f"{pred['P(H)']:.1%}")
        c2.metric("P(Empate)", f"{pred['P(D)']:.1%}")
        c3.metric("P(Visitante)", f"{pred['P(A)']:.1%}")

        st.markdown(f"**Goles esperados:** {pred['lambda_home']:.2f} — {pred['lambda_away']:.2f}")

        sm = pred["score_matrix"]
        idx = np.unravel_index(np.argmax(sm), sm.shape)
        over25 = 1 - sum(sm[i][j] for i in range(sm.shape[0]) for j in range(sm.shape[1]) if i + j <= 2)
        st.markdown(f"**Marcador más probable:** {idx[0]}-{idx[1]} ({sm[idx]:.1%})")
        st.markdown(f"**P(Over 2.5 goles):** {over25:.1%}")

        st.markdown("---")
        st.markdown("**Otras métricas esperadas**")
        cols = st.columns(len(metric_models))
        for col, (label, m) in zip(cols, metric_models.items()):
            if home in m.teams and away in m.teams:
                mp = m.predict_match(home, away)
                col.metric(label, f"{mp['expected_home']:.1f} — {mp['expected_away']:.1f}")

        # --- Comparador de cuotas ---
        st.markdown("---")
        st.markdown("**Comparar con cuota del mercado (opcional)**")
        oc1, oc2, oc3 = st.columns(3)
        odds_h = oc1.number_input("Cuota Local", min_value=1.01, value=2.00, step=0.01, key="odds_h")
        odds_d = oc2.number_input("Cuota Empate", min_value=1.01, value=3.30, step=0.01, key="odds_d")
        odds_a = oc3.number_input("Cuota Visitante", min_value=1.01, value=3.50, step=0.01, key="odds_a")

        raw_h, raw_d, raw_a = 1 / odds_h, 1 / odds_d, 1 / odds_a
        total_raw = raw_h + raw_d + raw_a
        implied_h, implied_d, implied_a = raw_h / total_raw, raw_d / total_raw, raw_a / total_raw

        edges = {
            "Local": pred["P(H)"] - implied_h,
            "Empate": pred["P(D)"] - implied_d,
            "Visitante": pred["P(A)"] - implied_a,
        }
        e1, e2, e3 = st.columns(3)
        for col, (label, edge) in zip([e1, e2, e3], edges.items()):
            if edge > 0.05:
                col.success(f"{label}\n\nEdge: {edge:+.1%}")
            elif edge < -0.05:
                col.error(f"{label}\n\nEdge: {edge:+.1%}")
            else:
                col.info(f"{label}\n\nEdge: {edge:+.1%}")
        st.caption(
            "Edge = probabilidad del modelo menos probabilidad implícita de la cuota. "
            "Recuerda: en nuestro propio backtest (pestaña 'Rendimiento histórico'), "
            "un edge positivo no ha garantizado beneficio real."
        )

        # --- Exportar ---
        st.markdown("---")
        html_report = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>{home} vs {away}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#f4f4f2; padding:32px; }}
.card {{ max-width:480px; margin:0 auto; background:#fff; border-radius:14px; padding:24px; }}
h1 {{ font-size:20px; }}
.row {{ display:flex; justify-content:space-between; margin:8px 0; }}
</style></head>
<body>
<div class="card">
<h1>{home} vs {away}</h1>
<p>{league_label}</p>
<div class="row"><span>Goles esperados</span><span>{pred['lambda_home']:.2f} — {pred['lambda_away']:.2f}</span></div>
<div class="row"><span>P(Local)</span><span>{pred['P(H)']:.1%}</span></div>
<div class="row"><span>P(Empate)</span><span>{pred['P(D)']:.1%}</span></div>
<div class="row"><span>P(Visitante)</span><span>{pred['P(A)']:.1%}</span></div>
<div class="row"><span>Marcador más probable</span><span>{idx[0]}-{idx[1]}</span></div>
<div class="row"><span>Over 2.5 goles</span><span>{over25:.1%}</span></div>
<p style="font-size:12px;color:#888;margin-top:16px;">
Modelo estadístico, no es una recomendación de apuesta.
</p>
</div>
</body></html>"""
        st.download_button(
            "Descargar esta predicción (HTML)",
            data=html_report,
            file_name=f"prediccion_{home}_vs_{away}.html".replace(" ", "_"),
            mime="text/html",
        )

# ---------------------------------------------------------------
# TAB 2: Ficha de equipo
# ---------------------------------------------------------------
with tab_team:
    team_choice = st.selectbox("Equipo", sorted(goals_model.teams), key="team_profile")

    n = len(goals_model.teams)
    idx_map = {t: i for i, t in enumerate(goals_model.teams)}
    attack_goals = goals_model.params[:n]
    defense_goals = goals_model.params[n:2 * n]
    i = idx_map[team_choice]

    elo_table = elo.current_table()
    elo_row = elo_table[elo_table["team"] == team_choice]
    elo_value = int(elo_row["elo"].iloc[0]) if not elo_row.empty else None
    elo_rank = int(elo_table.reset_index(drop=True).index[elo_table["team"] == team_choice][0]) + 1 if not elo_row.empty else None

    st.subheader(team_choice)
    c1, c2, c3 = st.columns(3)
    c1.metric("Elo actual", elo_value if elo_value else "—", help=f"Puesto #{elo_rank} de {len(elo_table)}" if elo_rank else None)
    c2.metric("Ataque (goles)", f"{attack_goals[i]:+.2f}")
    c3.metric("Defensa (goles)", f"{defense_goals[i]:+.2f}", help="Más bajo = mejor defensa (concede menos)")

    st.markdown("**Otras métricas (ataque / defensa)**")
    cols = st.columns(len(metric_models))
    for col, (label, m) in zip(cols, metric_models.items()):
        if team_choice in m.teams:
            ratings = m.team_ratings()
            row = ratings[ratings["team"] == team_choice].iloc[0]
            col.metric(label, f"{row[f'{label}_attack']:+.2f} / {row[f'{label}_defense']:+.2f}")

# ---------------------------------------------------------------
# TAB 3: Ranking Elo
# ---------------------------------------------------------------
with tab_elo:
    st.subheader("Ranking Elo actual")
    table = elo.current_table()
    table["elo"] = table["elo"].round(0).astype(int)
    st.dataframe(table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# TAB 4: Ratings por equipo (ataque/defensa)
# ---------------------------------------------------------------
with tab_ratings:
    metric_choice = st.selectbox("Métrica", list(metric_models.keys()))
    m = metric_models[metric_choice]
    ratings = m.team_ratings().reset_index(drop=True)
    ratings.columns = ["Equipo", "Ataque", "Defensa"]
    ratings["Ataque"] = ratings["Ataque"].round(2)
    ratings["Defensa"] = ratings["Defensa"].round(2)
    st.caption("Ataque más alto = genera más. Defensa más baja = concede menos.")
    st.dataframe(ratings, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# TAB 5: Estadísticas generales de la liga
# ---------------------------------------------------------------
with tab_stats:
    df["Season"] = df["Date"].apply(
        lambda d: f"{d.year}-{str(d.year + 1)[2:]}" if d.month >= 7 else f"{d.year - 1}-{str(d.year)[2:]}"
    )
    season_stats = df.groupby("Season").apply(
        lambda g: pd.Series({
            "Partidos": len(g),
            "Goles/partido": (g["FTHG"] + g["FTAG"]).mean().round(2),
            "% Local": (g["FTR"] == "H").mean() * 100,
            "% Empate": (g["FTR"] == "D").mean() * 100,
            "% Visitante": (g["FTR"] == "A").mean() * 100,
        })
    ).round(1)
    st.dataframe(season_stats, use_container_width=True)

    st.bar_chart(season_stats["Goles/partido"])

# ---------------------------------------------------------------
# TAB 6: Rendimiento histórico (backtest walk-forward)
# ---------------------------------------------------------------
with tab_backtest:
    st.subheader("Rendimiento histórico del modelo (backtest walk-forward)")
    st.caption(
        "Simula apostar solo cuando el modelo ve un edge de al menos 5% frente a la cuota media "
        "del mercado. Cada temporada se predice entrenando solo con las temporadas anteriores "
        "(el modelo nunca mira el futuro)."
    )

    bt = get_backtest(league_code, decay)

    if bt.empty:
        st.info("No hay histórico suficiente en esta liga para un backtest walk-forward fiable.")
    else:
        n_bets = len(bt)
        wins = bt["won"].sum()
        profit = bt["pnl"].sum()
        roi = profit / n_bets * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Apuestas simuladas", n_bets)
        c2.metric("Acierto", f"{wins / n_bets:.1%}")
        c3.metric("P&L", f"{profit:+.1f}u")
        c4.metric("ROI", f"{roi:+.1f}%")

        if roi < 0:
            st.warning(
                "ROI negativo: en este backtest, el modelo NO ha batido al mercado de forma "
                "consistente en esta liga. Trátalo como información, no como señal para apostar."
            )
        else:
            st.info(
                "ROI positivo en este backtest — pero con muestras moderadas, esto puede deberse "
                "al azar. No lo tomes como garantía de rentabilidad futura."
            )

        st.markdown("**Por temporada**")
        season_bt = bt.groupby("season").apply(
            lambda g: pd.Series({
                "Apuestas": len(g),
                "Acierto %": g["won"].mean() * 100,
                "ROI %": g["pnl"].sum() / len(g) * 100,
            })
        ).round(1)
        st.dataframe(season_bt, use_container_width=True)
        st.bar_chart(season_bt["ROI %"])
