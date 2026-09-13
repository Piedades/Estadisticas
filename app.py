"""
Dashboard interactivo — Modelo predictivo Big 5 Ligas Europeas
Ejecutar con: streamlit run app.py
"""
import glob
import pandas as pd
import streamlit as st

from datetime import date

from data_pipeline_and_model import DixonColesModel, load_league_data
from generic_poisson_model import GenericPoissonModel
from elo_ratings import EloTracker
from fetch_fixtures import fetch_matches_for_date, resolve_team_name, load_aliases

st.set_page_config(page_title="Big 5 Ligas — Panel de predicción", layout="wide")

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

df = get_league_df(league_code)
goals_model = get_goals_model(league_code, decay)
metric_models = get_metric_models(league_code)
elo = get_elo(league_code)

st.title(f"{league_label}")
st.caption(f"{len(df)} partidos cargados · {df['Date'].min().date()} a {df['Date'].max().date()}")

tab_today, tab_pred, tab_elo, tab_ratings, tab_stats = st.tabs(
    ["📅 Partidos del día", "🔮 Predecir partido", "📊 Ranking Elo", "🎯 Ratings por equipo", "📈 Estadísticas de la liga"]
)

# ---------------------------------------------------------------
# TAB 0: Partidos del día (autodetección desde besoccer.es)
# ---------------------------------------------------------------
with tab_today:
    st.subheader("Buscar partidos por fecha")
    st.caption(
        "Trae los partidos programados de besoccer.es para el día elegido, "
        "filtra las 5 grandes ligas y calcula la predicción de cada uno."
    )

    picked_date = st.date_input("Fecha", value=date.today())
    buscar = st.button("Buscar partidos", type="primary")

    if buscar:
        date_str = picked_date.strftime("%Y-%m-%d")
        try:
            with st.spinner(f"Consultando besoccer.es para el {date_str}..."):
                raw_matches = fetch_matches_for_date(date_str)
        except Exception as e:
            st.error(
                f"No se ha podido consultar besoccer.es: {e}\n\n"
                "Revisa tu conexión a internet. Si el error persiste, puede que "
                "besoccer.es haya cambiado su estructura — el HTML de la última "
                "consulta se guardó en `last_fetch.html` para poder revisarlo."
            )
            raw_matches = []

        if not raw_matches:
            st.info("No se han encontrado partidos de las 5 grandes ligas para esa fecha.")
        else:
            aliases = load_aliases()
            st.success(f"Encontrados {len(raw_matches)} partidos de las 5 ligas.")

            for m in raw_matches:
                lg = m["league_code"]
                model = get_goals_model(lg, decay)
                known_teams = list(model.teams)

                home = resolve_team_name(m["home_raw"], lg, known_teams, aliases)
                away = resolve_team_name(m["away_raw"], lg, known_teams, aliases)

                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{LEAGUES[lg]}** · {m['time']}")

                    if home is None or away is None:
                        st.warning(
                            f"No he podido emparejar: '{m['home_raw']}' vs '{m['away_raw']}' "
                            f"con ningún equipo conocido de {lg}. "
                            f"Añádelo a team_aliases.json y vuelve a buscar."
                        )
                        continue

                    pred = model.predict_match(home, away)
                    st.markdown(f"### {home}  {pred['lambda_home']:.2f} — {pred['lambda_away']:.2f}  {away}")

                    p1, p2, p3 = st.columns(3)
                    p1.metric(f"Gana {home}", f"{pred['P(H)']:.1%}")
                    p2.metric("Empate", f"{pred['P(D)']:.1%}")
                    p3.metric(f"Gana {away}", f"{pred['P(A)']:.1%}")

# ---------------------------------------------------------------
# TAB 1: Predicción de partido
# ---------------------------------------------------------------
with tab_pred:
    teams = sorted(goals_model.teams)
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("Equipo local", teams, index=0)
    with col2:
        away_options = [t for t in teams if t != home]
        away = st.selectbox("Equipo visitante", away_options, index=0)

    if st.button("Calcular predicción", type="primary"):
        pred = goals_model.predict_match(home, away)

        st.subheader(f"{home} vs {away}")

        c1, c2, c3 = st.columns(3)
        c1.metric("P(Local)", f"{pred['P(H)']:.1%}")
        c2.metric("P(Empate)", f"{pred['P(D)']:.1%}")
        c3.metric("P(Visitante)", f"{pred['P(A)']:.1%}")

        st.markdown(f"**Goles esperados:** {pred['lambda_home']:.2f} — {pred['lambda_away']:.2f}")

        import numpy as np
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

# ---------------------------------------------------------------
# TAB 2: Ranking Elo
# ---------------------------------------------------------------
with tab_elo:
    st.subheader("Ranking Elo actual")
    table = elo.current_table()
    table["elo"] = table["elo"].round(0).astype(int)
    st.dataframe(table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# TAB 3: Ratings por equipo (ataque/defensa)
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
# TAB 4: Estadísticas generales de la liga
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
