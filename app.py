"""
Dashboard interactivo — Modelo predictivo Big 5 Ligas Europeas
Ejecutar con: streamlit run app.py
"""
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from data_pipeline_and_model import DixonColesModel, load_league_data
from generic_poisson_model import GenericPoissonModel
from elo_ratings import EloTracker
from backtest import run_walk_forward, run_calibration
from github_sync import get_file, put_file
from auto_update import update_league, current_season_code
from auth import authenticate, register_user, load_users, is_admin, delete_user, update_invite_code
from season_sim import simulate_season
from team_stats import team_match_averages, METRIC_COLUMNS
from fixtures_api import fetch_matches_for_date, resolve_team_name
from visuals import render_prob_bar, render_mini_prob_bar, favorite_badge, render_form_badges

st.set_page_config(page_title="Big 5 Ligas — Panel de predicción", layout="wide")

# ---------------------------------------------------------------
# Toques de estilo extra (ademas del tema en .streamlit/config.toml)
# ---------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stVerticalBlockBorderWrapper"] > div {
border-radius: 14px !important;
box-shadow: 0 2px 10px rgba(0,0,0,0.08);
transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
transform: translateY(-2px);
box-shadow: 0 6px 18px rgba(0,0,0,0.14);
}
[data-testid="stMetric"] {
background: rgba(0,0,0,0.03);
border-radius: 12px;
padding: 12px 14px;
border: 1px solid rgba(0,0,0,0.08);
}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
border-radius: 10px !important;
font-weight: 600 !important;
}
[data-testid="stTabs"] button {
border-radius: 8px 8px 0 0;
}
[data-testid="stSidebar"] {
background: linear-gradient(180deg, #eef1f6 0%, #e3e7ee 100%);
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Acceso con usuarios individuales
# ---------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

if not st.session_state["authenticated"]:
    st.title("⚽ Big 5 Ligas")
    st.caption("🔒 Acceso privado — inicia sesión o crea tu cuenta para entrar.")

    login_tab, register_tab = st.tabs(["Iniciar sesión", "Crear cuenta"])

    with login_tab:
        with st.form("login_form"):
            login_user = st.text_input("Usuario")
            login_pwd = st.text_input("Contraseña", type="password")
            login_submit = st.form_submit_button("Entrar")
        if login_submit:
            if authenticate(login_user, login_pwd):
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = login_user.strip().lower()
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

    with register_tab:
        st.caption("Necesitas el código de invitación para crear una cuenta nueva.")
        with st.form("register_form"):
            new_user = st.text_input("Elige un usuario")
            new_pwd = st.text_input("Elige una contraseña (mínimo 6 caracteres)", type="password")
            invite_code = st.text_input("Código de invitación", type="password")
            register_submit = st.form_submit_button("Crear cuenta")
        if register_submit:
            ok, msg = register_user(new_user, new_pwd, invite_code)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

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


@st.cache_data(show_spinner="Calculando calibración del modelo...")
def get_calibration(league_code: str, decay: float) -> pd.DataFrame:
    df = get_league_df(league_code)
    return run_calibration(df, decay=decay)


@st.cache_data(show_spinner="Simulando el resto de la temporada (puede tardar unos segundos)...")
def get_season_sim(league_code: str, decay: float, season: str, top_europe: int, relegation: int) -> pd.DataFrame:
    df = get_league_df(league_code)
    return simulate_season(df, season, decay=decay, n_sims=1500, top_europe=top_europe, relegation=relegation)


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("⚽ Big 5 Ligas")
league_label = st.sidebar.selectbox("Liga", list(LEAGUES.values()), key="league_select")
league_code = [k for k, v in LEAGUES.items() if v == league_label][0]

half_life_months = st.sidebar.slider(
    "Vida media de los partidos antiguos (meses)", 4, 46, 15,
    help=(
        "Cuántos meses tarda un partido en pesar la mitad en el modelo. "
        "Más bajo = el modelo se adapta más rápido a la forma reciente. "
        "Más alto = usa más historial, más lento para reaccionar a rachas."
    )
)
decay = math.log(2) / (half_life_months * 30)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Modelo Poisson / Dixon-Coles entrenado con tus datos locales de football-data.co.uk. "
    "Nada de esto sale de tu ordenador."
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Sesión: **{st.session_state['current_user']}**")
if st.sidebar.button("Cerrar sesión"):
    st.session_state["authenticated"] = False
    st.session_state["current_user"] = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Actualizar datos")
st.sidebar.caption(f"Temporada en curso: {current_season_code()}")

if st.sidebar.button(f"🔄 Actualizar {LEAGUES[league_code]} ahora"):
    with st.spinner(f"Descargando datos de {LEAGUES[league_code]} desde football-data.co.uk..."):
        ok, msg = update_league(league_code)
    if ok:
        st.sidebar.success(msg + " Tardará 1-2 minutos en desplegarse; luego recarga la página.")
        get_league_df.clear()
    else:
        st.sidebar.error(msg)

if st.sidebar.button("🔄 Actualizar las 5 grandes ligas"):
    results = []
    progress = st.sidebar.progress(0.0)
    for i, (code, name) in enumerate(LEAGUES.items()):
        with st.spinner(f"Descargando datos de {name} desde football-data.co.uk..."):
            ok_i, msg_i = update_league(code)
            results.append((name, ok_i, msg_i))
        progress.progress((i + 1) / len(LEAGUES))
    progress.empty()
    for name, ok_i, msg_i in results:
        if ok_i:
            st.sidebar.success(f"{name}: {msg_i}")
        else:
            st.sidebar.error(f"{name}: {msg_i}")
    get_league_df.clear()
    st.sidebar.info("Tardará 1-2 minutos en desplegarse; luego recarga la página.")

with st.sidebar.expander("Subir CSV manualmente (alternativa)"):
    uploaded_csv = st.file_uploader("Subir CSV de una temporada nueva", type="csv")
    if uploaded_csv is not None:
        upload_league = st.selectbox(
            "¿A qué liga pertenece?", list(LEAGUES.keys()),
            format_func=lambda k: LEAGUES[k], key="upload_league_select"
        )
        if st.button("Guardar en GitHub", key="manual_upload_btn"):
            csv_text = uploaded_csv.getvalue().decode("utf-8", errors="replace")
            ok, msg = put_file(
                f"data/{uploaded_csv.name}", csv_text,
                f"Subir {uploaded_csv.name} ({LEAGUES[upload_league]}) desde la app"
            )
            if ok:
                st.success(msg + " Tardará 1-2 minutos en desplegarse; luego recarga la página.")
                get_league_df.clear()
            else:
                st.error(msg)

df = get_league_df(league_code)
goals_model = get_goals_model(league_code, decay)
metric_models = get_metric_models(league_code)
elo = get_elo(league_code)

st.title(f"{league_label}")
st.caption(f"{len(df)} partidos cargados · {df['Date'].min().date()} a {df['Date'].max().date()}")

tab_defs = [
    ("📅 Partidos del día", "today"),
    ("🔮 Predecir partido", "pred"),
    ("🗂️ Ficha de equipo", "team"),
    ("⚖️ Comparar equipos", "compare"),
    ("📊 Ranking Elo", "elo"),
    ("🎯 Ratings por equipo", "ratings"),
    ("📈 Estadísticas de la liga", "stats"),
    ("🎲 Simulador de temporada", "sim"),
    ("🗄️ Más", "more"),
]
current_is_admin = is_admin(st.session_state["current_user"])

tab_objects = st.tabs([label for label, _ in tab_defs])
tabs = {key: tab for (label, key), tab in zip(tab_defs, tab_objects)}
tab_today = tabs["today"]
tab_pred = tabs["pred"]
tab_team = tabs["team"]
tab_compare = tabs["compare"]
tab_elo = tabs["elo"]
tab_ratings = tabs["ratings"]
tab_stats = tabs["stats"]
tab_sim = tabs["sim"]
tab_more = tabs["more"]

with tab_more:
    more_defs = [("🧪 Rendimiento histórico", "backtest"), ("📓 Diario de apuestas", "diary")]
    if current_is_admin:
        more_defs.append(("🛠️ Administración", "admin"))
    more_objects = st.tabs([label for label, _ in more_defs])
    more_tabs = {key: tab for (label, key), tab in zip(more_defs, more_objects)}
    tab_backtest = more_tabs["backtest"]
    tab_diary = more_tabs["diary"]
    tab_admin = more_tabs.get("admin")

# ---------------------------------------------------------------
# TAB: Partidos del día (calendario vía football-data.org)
# ---------------------------------------------------------------
with tab_today:
    st.subheader("Partidos del día")
    st.caption(
        "Calendario en vivo de las 5 grandes ligas, vía football-data.org "
        "(no besoccer.es). Necesita el secret FOOTBALL_DATA_TOKEN configurado."
    )

    def _select_match(lg, home, away):
        st.session_state["league_select"] = LEAGUES[lg]
        st.session_state["home_team"] = home
        st.session_state["away_team"] = away
        st.session_state["show_prediction"] = True

    if st.session_state.get("show_prediction") and st.session_state.get("home_team"):
        st.success(
            f"✅ Predicción lista para **{st.session_state['home_team']} vs "
            f"{st.session_state['away_team']}** — ve a la pestaña "
            f"**\"🔮 Predecir partido\"** para verla completa (ya está todo calculado)."
        )

    import datetime as _dt
    picked_date = st.date_input("Fecha", value=_dt.date.today(), key="today_date")

    if st.button("Buscar partidos", type="primary", key="today_search_btn"):
        with st.spinner("Consultando football-data.org..."):
            matches, err = fetch_matches_for_date(picked_date.strftime("%Y-%m-%d"))
        st.session_state["today_matches"] = matches
        st.session_state["today_error"] = err

    if st.session_state.get("today_error"):
        st.error(st.session_state["today_error"])

    matches_today = st.session_state.get("today_matches")
    if matches_today is not None:
        if not matches_today:
            st.info("No hay partidos programados de las 5 grandes ligas para esa fecha.")
        else:
            st.success(f"{len(matches_today)} partidos encontrados.")
            for m in matches_today:
                lg = m["league_code"]
                league_teams = sorted(get_goals_model(lg, decay).teams)
                home = resolve_team_name(m["home_raw"], league_teams)
                away = resolve_team_name(m["away_raw"], league_teams)

                with st.container(border=True):
                    top1, top2 = st.columns([4, 1])
                    top1.markdown(f"**{LEAGUES[lg]}**")
                    top2.markdown(f"🕒 {m['time']}")

                    if home is None or away is None:
                        st.markdown(f"**{m['home_raw']}** vs **{m['away_raw']}**")
                        st.caption(
                            "No he podido emparejar uno de estos equipos con nuestros datos "
                            "históricos (nombre distinto). No se puede calcular predicción."
                        )
                        continue

                    model_today = get_goals_model(lg, decay)
                    pred_today = model_today.predict_match(home, away)
                    st.markdown(f"### {home}  vs  {away}")
                    render_mini_prob_bar(pred_today["P(H)"], pred_today["P(D)"], pred_today["P(A)"])
                    st.caption(favorite_badge(pred_today["P(H)"], pred_today["P(D)"], pred_today["P(A)"], home, away))

                    st.button(
                        "Ver predicción completa →",
                        key=f"today_predict_{lg}_{home}_{away}",
                        on_click=_select_match, args=(lg, home, away),
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
        render_prob_bar(pred["P(H)"], pred["P(D)"], pred["P(A)"], home, away)

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

        # --- Over/Under ---
        st.markdown("---")
        st.markdown("**Probabilidades Over/Under**")
        default_lines = {"Córners": 9.5, "Tiros": 24.5, "Tiros a puerta": 8.5, "Tarjetas amarillas": 3.5}
        ou_cols = st.columns(len(metric_models))
        for col, (label, m) in zip(ou_cols, metric_models.items()):
            if home in m.teams and away in m.teams:
                mp = m.predict_match(home, away)
                total_mu = mp["expected_home"] + mp["expected_away"]
                line = col.number_input(
                    f"Línea {label}", value=default_lines.get(label, 3.5), step=0.5, key=f"ou_line_{label}"
                )
                p_over = 1 - poisson.cdf(line, total_mu)
                col.metric(f"P(Over {line})", f"{p_over:.1%}")

        # --- Historial cara a cara ---
        st.markdown("---")
        st.markdown("**Historial cara a cara**")
        h2h = df[
            ((df["HomeTeam"] == home) & (df["AwayTeam"] == away))
            | ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
        ].sort_values("Date", ascending=False).head(5)
        if h2h.empty:
            st.caption("No hay enfrentamientos previos entre estos dos equipos en los datos cargados.")
        else:
            h2h_display = h2h[["Date", "HomeTeam", "FTHG", "FTAG", "AwayTeam"]].copy()
            h2h_display["Date"] = h2h_display["Date"].dt.date
            h2h_display.columns = ["Fecha", "Local", "Goles Local", "Goles Visitante", "Visitante"]
            st.dataframe(h2h_display, use_container_width=True, hide_index=True)

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
            "📥 Descargar esta predicción (HTML)",
            data=html_report,
            file_name=f"prediccion_{home}_vs_{away}.html".replace(" ", "_"),
            mime="text/html",
        )

# ---------------------------------------------------------------
# TAB 2: Ficha de equipo
# ---------------------------------------------------------------
with tab_team:
    team_choice = st.selectbox("Equipo", sorted(goals_model.teams), key="team_profile")

    elo_table = elo.current_table()
    elo_row = elo_table[elo_table["team"] == team_choice]
    elo_value = int(elo_row["elo"].iloc[0]) if not elo_row.empty else None
    elo_rank = int(elo_table.reset_index(drop=True).index[elo_table["team"] == team_choice][0]) + 1 if not elo_row.empty else None

    avg = team_match_averages(df, team_choice)

    st.subheader(team_choice)
    c1, c2, c3 = st.columns(3)
    c1.metric("Elo actual", elo_value if elo_value else "—", help=f"Puesto #{elo_rank} de {len(elo_table)}" if elo_rank else None)
    c2.metric("Goles a favor (por partido)", f"{avg.get('Goles a favor', 0):.2f}")
    c3.metric("Goles en contra (por partido)", f"{avg.get('Goles en contra', 0):.2f}")
    st.caption(f"Sobre {avg.get('Partidos jugados', 0)} partidos jugados en los datos cargados.")

    st.markdown("**Otras métricas por partido (a favor / en contra)**")
    cols = st.columns(len(METRIC_COLUMNS))
    for col, label in zip(cols, METRIC_COLUMNS.keys()):
        favor = avg.get(f"{label} a favor")
        contra = avg.get(f"{label} en contra")
        if favor is not None:
            col.metric(label, f"{favor:.1f} / {contra:.1f}")

    st.markdown("---")
    st.markdown("**Forma reciente (últimos 5 partidos)**")
    team_matches = df[
        (df["HomeTeam"] == team_choice) | (df["AwayTeam"] == team_choice)
    ].sort_values("Date", ascending=False).head(5).sort_values("Date")
    if team_matches.empty:
        st.caption("No hay partidos recientes en los datos cargados.")
    else:
        letters = []
        for _, row in team_matches.iterrows():
            if row["HomeTeam"] == team_choice:
                gf, gc = row["FTHG"], row["FTAG"]
            else:
                gf, gc = row["FTAG"], row["FTHG"]
            letters.append("V" if gf > gc else "E" if gf == gc else "D")
        puntos = sum(3 if r == "V" else 1 if r == "E" else 0 for r in letters)
        render_form_badges(letters)
        st.caption(f"{puntos} de {len(letters) * 3} puntos posibles en los últimos {len(letters)} partidos.")

    st.markdown("---")
    st.markdown("**Evolución del Elo**")
    elo_hist = elo.team_elo_history(team_choice)
    if elo_hist.empty:
        st.caption("No hay histórico suficiente para graficar la evolución.")
    else:
        st.line_chart(elo_hist.set_index("date")["elo"])

# ---------------------------------------------------------------
# TAB: Comparar equipos
# ---------------------------------------------------------------
with tab_compare:
    cc1, cc2 = st.columns(2)
    team_options = sorted(goals_model.teams)
    with cc1:
        team_a = st.selectbox("Equipo A", team_options, index=0, key="compare_a")
    with cc2:
        team_b_options = [t for t in team_options if t != team_a]
        team_b = st.selectbox("Equipo B", team_b_options, index=0, key="compare_b")

    elo_table_cmp = elo.current_table().set_index("team")["elo"]

    def _team_row(team):
        avg = team_match_averages(df, team)
        row = {
            "Elo": f"{elo_table_cmp.get(team, 0):.0f}",
            "Goles a favor": f"{avg.get('Goles a favor', 0):.2f}",
            "Goles en contra": f"{avg.get('Goles en contra', 0):.2f}",
        }
        for label in METRIC_COLUMNS.keys():
            favor = avg.get(f"{label} a favor")
            contra = avg.get(f"{label} en contra")
            if favor is not None:
                row[label] = f"{favor:.1f} / {contra:.1f}"
        tm = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date", ascending=False).head(5).sort_values("Date")
        letters = []
        for _, r in tm.iterrows():
            gf, gc = (r["FTHG"], r["FTAG"]) if r["HomeTeam"] == team else (r["FTAG"], r["FTHG"])
            letters.append("V" if gf > gc else "E" if gf == gc else "D")
        row["Forma reciente"] = "  ".join(letters) if letters else "—"
        return row

    row_a = _team_row(team_a)
    row_b = _team_row(team_b)
    compare_df = pd.DataFrame({team_a: row_a, team_b: row_b})
    st.dataframe(compare_df, use_container_width=True)
    st.caption("Córners / Tiros / Tiros a puerta / Tarjetas amarillas en formato \"a favor / en contra\", promedio por partido.")

    if st.button("Ver predicción de este partido", key="compare_predict_btn"):
        pred_cmp = goals_model.predict_match(team_a, team_b)
        st.markdown(f"**{team_a} {pred_cmp['lambda_home']:.2f} — {pred_cmp['lambda_away']:.2f} {team_b}**")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric(f"Gana {team_a}", f"{pred_cmp['P(H)']:.1%}")
        pc2.metric("Empate", f"{pred_cmp['P(D)']:.1%}")
        pc3.metric(f"Gana {team_b}", f"{pred_cmp['P(A)']:.1%}")
        render_prob_bar(pred_cmp["P(H)"], pred_cmp["P(D)"], pred_cmp["P(A)"], team_a, team_b)

# ---------------------------------------------------------------
# TAB 3: Ranking Elo
# ---------------------------------------------------------------
with tab_elo:
    st.subheader("📊 Ranking Elo actual")
    table = elo.current_table()
    table["elo"] = table["elo"].round(0).astype(int)
    table = table.reset_index(drop=True)
    medals = ["🥇", "🥈", "🥉"] + [str(i) for i in range(4, len(table) + 1)]
    table.insert(0, "Pos", medals[:len(table)])
    table.columns = ["Pos", "Equipo", "Elo"]
    styled_elo = table.style.background_gradient(subset=["Elo"], cmap="RdYlGn")
    st.dataframe(styled_elo, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# TAB 4: Ratings por equipo (ataque/defensa)
# ---------------------------------------------------------------
with tab_ratings:
    metric_choice = st.selectbox("Métrica", ["Goles"] + list(METRIC_COLUMNS.keys()))
    rows = []
    for team in sorted(goals_model.teams):
        avg = team_match_averages(df, team)
        if metric_choice == "Goles":
            favor = avg.get("Goles a favor")
            contra = avg.get("Goles en contra")
        else:
            favor = avg.get(f"{metric_choice} a favor")
            contra = avg.get(f"{metric_choice} en contra")
        if favor is None:
            continue
        rows.append({
            "Equipo": team,
            "A favor (por partido)": round(favor, 2),
            "En contra (por partido)": round(contra, 2),
        })
    ratings = pd.DataFrame(rows).sort_values("A favor (por partido)", ascending=False).reset_index(drop=True)
    st.caption("🟢 Promedio real por partido. Más alto en \"a favor\" = genera más. Más bajo en \"en contra\" = concede menos (más verde).")
    styled_ratings = (
        ratings.style
        .format({"A favor (por partido)": "{:.2f}", "En contra (por partido)": "{:.2f}"})
        .background_gradient(subset=["A favor (por partido)"], cmap="Greens")
        .background_gradient(subset=["En contra (por partido)"], cmap="RdYlGn_r")
    )
    st.dataframe(styled_ratings, use_container_width=True, hide_index=True)

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
    styled_season = season_stats.style.format({
        "Goles/partido": "{:.2f}", "% Local": "{:.1f}", "% Empate": "{:.1f}", "% Visitante": "{:.1f}",
    }).background_gradient(subset=["Goles/partido"], cmap="YlOrRd")
    st.dataframe(styled_season, use_container_width=True)

    st.bar_chart(season_stats["Goles/partido"])

# ---------------------------------------------------------------
# TAB: Simulador de temporada completa (Monte Carlo)
# ---------------------------------------------------------------
with tab_sim:
    st.subheader("Simulador de temporada completa")
    st.caption(
        "Simula todos los partidos que quedan de la temporada en curso, 1.500 veces seguidas, "
        "para calcular la probabilidad de cada equipo de ser campeón, entrar en puestos europeos "
        "o descender. Como no tenemos el calendario real de partidos futuros, se asume que cada "
        "equipo juega dos veces contra cada rival (una en casa, otra fuera) y se descuentan los "
        "partidos que ya se han jugado. Es una aproximación, no el calendario exacto."
    )

    sc1, sc2 = st.columns(2)
    top_europe_n = sc1.number_input("Nº de puestos europeos a contar", min_value=1, max_value=8, value=4)
    relegation_n = sc2.number_input("Nº de puestos de descenso a contar", min_value=0, max_value=6, value=3)

    current_season_data = sorted(df["Season"].dropna().unique())[-1]
    st.caption(f"Simulando el resto de la temporada {current_season_data}.")

    if st.button("Simular temporada", type="primary"):
        sim_result = get_season_sim(league_code, decay, current_season_data, int(top_europe_n), int(relegation_n))
        st.session_state["sim_result"] = sim_result

    if "sim_result" in st.session_state:
        sim_display = st.session_state["sim_result"].copy()
        sim_display["P(Campeón)"] = (sim_display["P(Campeón)"] * 100).round(1)
        sim_display["P(Puestos europeos)"] = (sim_display["P(Puestos europeos)"] * 100).round(1)
        sim_display["P(Descenso)"] = (sim_display["P(Descenso)"] * 100).round(1)
        sim_display.columns = ["Equipo", "Puntos actuales", "P(Campeón) %", "P(Puestos europeos) %", "P(Descenso) %"]
        styled_sim = (
            sim_display.style
            .format({"P(Campeón) %": "{:.1f}", "P(Puestos europeos) %": "{:.1f}", "P(Descenso) %": "{:.1f}"})
            .background_gradient(subset=["P(Campeón) %"], cmap="YlOrRd")
            .background_gradient(subset=["P(Puestos europeos) %"], cmap="Blues")
            .background_gradient(subset=["P(Descenso) %"], cmap="Reds")
        )
        st.dataframe(styled_sim, use_container_width=True, hide_index=True)
        st.bar_chart(sim_display.set_index("Equipo")["P(Campeón) %"])

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
                "⚠️ ROI negativo: en este backtest, el modelo NO ha batido al mercado de forma "
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

    st.markdown("---")
    st.subheader("Calibración del modelo")
    st.caption(
        "Agrupa TODAS las predicciones (no solo las apuestas simuladas) según la probabilidad "
        "que el modelo asignó, y compara contra la frecuencia real de acierto en cada grupo. "
        "Si el modelo estuviera perfectamente calibrado, la probabilidad media y la frecuencia "
        "real deberían coincidir en cada fila."
    )
    calib = get_calibration(league_code, decay)
    if calib.empty:
        st.info("No hay histórico suficiente para calcular la calibración.")
    else:
        calib_display = calib.copy()
        calib_display["predicted_avg"] = (calib_display["predicted_avg"] * 100).round(1)
        calib_display["actual_freq"] = (calib_display["actual_freq"] * 100).round(1)
        calib_display.columns = ["Prob. media del modelo (%)", "Frecuencia real de acierto (%)", "Nº predicciones"]
        st.dataframe(calib_display, use_container_width=True, hide_index=True)
        chart_data = calib.set_index("predicted_avg")[["actual_freq"]]
        chart_data.columns = ["Frecuencia real"]
        st.line_chart(chart_data)

# ---------------------------------------------------------------
# TAB 7: Diario de apuestas propio
# ---------------------------------------------------------------
with tab_diary:
    st.subheader("Diario de apuestas propio")
    st.caption(
        "Registra aquí las apuestas que TÚ haces de verdad (no las simuladas del backtest), "
        "para comparar tu rendimiento real con el modelo. Solo ves las tuyas, aunque otros "
        "usuarios usen la misma app."
    )

    current_user = st.session_state["current_user"]

    log_content, _ = get_file("bets_log.csv")
    if log_content:
        all_bets_df = pd.read_csv(io.StringIO(log_content))
        if "usuario" not in all_bets_df.columns:
            all_bets_df["usuario"] = "desconocido"
    else:
        all_bets_df = pd.DataFrame(columns=[
            "usuario", "fecha", "liga", "local", "visitante", "seleccion", "cuota", "stake", "resultado"
        ])

    log_df = all_bets_df[all_bets_df["usuario"] == current_user].copy()

    with st.form("nueva_apuesta"):
        c1, c2, c3 = st.columns(3)
        f_date = c1.date_input("Fecha")
        f_liga = c2.selectbox("Liga", list(LEAGUES.values()), key="diary_liga")
        f_stake = c3.number_input("Stake (unidades)", min_value=0.0, value=1.0, step=0.5)

        c4, c5, c6 = st.columns(3)
        f_local = c4.text_input("Equipo local")
        f_visitante = c5.text_input("Equipo visitante")
        f_seleccion = c6.selectbox("Selección", ["Local", "Empate", "Visitante"])

        c7, c8 = st.columns(2)
        f_cuota = c7.number_input("Cuota", min_value=1.01, value=2.00, step=0.01)
        f_resultado = c8.selectbox("Resultado", ["Pendiente", "Ganada", "Perdida"])

        submitted = st.form_submit_button("Guardar apuesta")

    if submitted:
        new_row = pd.DataFrame([{
            "usuario": current_user,
            "fecha": f_date, "liga": f_liga, "local": f_local, "visitante": f_visitante,
            "seleccion": f_seleccion, "cuota": f_cuota, "stake": f_stake, "resultado": f_resultado,
        }])
        all_bets_df = pd.concat([all_bets_df, new_row], ignore_index=True)
        ok, msg = put_file("bets_log.csv", all_bets_df.to_csv(index=False), f"Añadir apuesta de {current_user}")
        if ok:
            st.success("Apuesta guardada.")
        else:
            st.error(msg)

    if not log_df.empty:
        st.markdown("---")
        st.markdown("**Tu historial de apuestas**")

        def _color_resultado(val):
            if val == "Ganada":
                return "background-color: #14532d; color: white; font-weight: 600;"
            elif val == "Perdida":
                return "background-color: #7f1d1d; color: white; font-weight: 600;"
            return "background-color: #78716c; color: white;"

        log_display = log_df.drop(columns=["usuario"])
        styled_log = log_display.style.format({"cuota": "{:.2f}", "stake": "{:.2f}"}).map(_color_resultado, subset=["resultado"])
        st.dataframe(styled_log, use_container_width=True, hide_index=True)

        resolved = log_df[log_df["resultado"].isin(["Ganada", "Perdida"])].copy()
        if not resolved.empty:
            resolved["pnl"] = resolved.apply(
                lambda r: r["stake"] * (r["cuota"] - 1) if r["resultado"] == "Ganada" else -r["stake"],
                axis=1,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Apuestas resueltas", len(resolved))
            c2.metric("P&L real", f"{resolved['pnl'].sum():+.1f}u")
            c3.metric("ROI real", f"{resolved['pnl'].sum() / resolved['stake'].sum():+.1%}")
    else:
        st.info("Todavía no has registrado ninguna apuesta.")

# ---------------------------------------------------------------
# TAB: Administración (solo visible para usuarios admin)
# ---------------------------------------------------------------
if tab_admin is not None:
    with tab_admin:
        st.subheader("Administración de usuarios")
        st.caption("Solo tú (como administrador) ves esta pestaña.")

        users = load_users()
        st.markdown(f"**{len(users)} usuarios registrados**")
        if users:
            st.dataframe(
                pd.DataFrame({"usuario": list(users.keys())}),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")
        st.markdown("**Borrar un usuario**")
        if users:
            user_to_delete = st.selectbox("Usuario a borrar", list(users.keys()), key="admin_delete_select")
            if st.button("Borrar cuenta", key="admin_delete_btn"):
                ok, msg = delete_user(user_to_delete)
                if ok:
                    st.success(f"Usuario '{user_to_delete}' borrado.")
                else:
                    st.error(msg)
        else:
            st.caption("No hay usuarios que borrar todavía.")

        st.markdown("---")
        st.markdown("**Cambiar el código de invitación**")
        st.caption(
            "Esto no toca los Secrets de Streamlit Cloud — se guarda en tu repositorio y "
            "sustituye al código anterior de inmediato."
        )
        new_invite = st.text_input("Nuevo código de invitación", key="admin_invite_input")
        if st.button("Actualizar código", key="admin_invite_btn"):
            ok, msg = update_invite_code(new_invite)
            if ok:
                st.success("Código de invitación actualizado.")
            else:
                st.error(msg)
