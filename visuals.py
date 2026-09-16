"""
Componentes visuales reutilizables para hacer la app más fácil de leer de
un vistazo: barras de probabilidad de colores en vez de solo números.
"""
import streamlit as st


def render_prob_bar(p_home: float, p_draw: float, p_away: float, label_home: str, label_away: str):
    """Barra horizontal de 3 colores mostrando P(Local)/P(Empate)/P(Visitante)."""
    h_pct, d_pct, a_pct = p_home * 100, p_draw * 100, p_away * 100
    html = f"""
    <div style="display:flex; height:40px; border-radius:10px; overflow:hidden;
                font-size:14px; color:white; font-weight:700; margin-top:6px;">
      <div style="width:{h_pct}%; background:#3b82f6; display:flex;
                  align-items:center; justify-content:center; min-width:32px;">
        {p_home:.0%}
      </div>
      <div style="width:{d_pct}%; background:#6b7280; display:flex;
                  align-items:center; justify-content:center; min-width:0;">
        {p_draw:.0%}
      </div>
      <div style="width:{a_pct}%; background:#ef4444; display:flex;
                  align-items:center; justify-content:center; min-width:32px;">
        {p_away:.0%}
      </div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:12px;
                color:#9ca3af; margin-top:4px; margin-bottom:12px;">
      <span>🔵 {label_home}</span><span>⚪ Empate</span><span>🔴 {label_away}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_mini_prob_bar(p_home: float, p_draw: float, p_away: float):
    """Versión compacta de la barra, para listas de varios partidos."""
    h_pct, d_pct, a_pct = p_home * 100, p_draw * 100, p_away * 100
    html = f"""
    <div style="display:flex; height:22px; border-radius:6px; overflow:hidden;
                font-size:11px; color:white; font-weight:600;">
      <div style="width:{h_pct}%; background:#3b82f6; display:flex;
                  align-items:center; justify-content:center;">{p_home:.0%}</div>
      <div style="width:{d_pct}%; background:#6b7280; display:flex;
                  align-items:center; justify-content:center;">{p_draw:.0%}</div>
      <div style="width:{a_pct}%; background:#ef4444; display:flex;
                  align-items:center; justify-content:center;">{p_away:.0%}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def favorite_badge(p_home: float, p_draw: float, p_away: float, home: str, away: str) -> str:
    """Devuelve un texto corto tipo '🔵 Favorito: Barcelona (62%)'."""
    best = max(("H", p_home), ("D", p_draw), ("A", p_away), key=lambda x: x[1])
    if best[0] == "H":
        return f"🔵 Favorito: {home} ({best[1]:.0%})"
    elif best[0] == "A":
        return f"🔴 Favorito: {away} ({best[1]:.0%})"
    return f"⚪ Partido muy igualado (empate {best[1]:.0%})"


def render_form_badges(letters: list):
    """Insignias de colores para la forma reciente: V verde, E gris, D roja."""
    colors = {"V": "#22c55e", "E": "#6b7280", "D": "#ef4444"}
    spans = "".join(
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:30px;height:30px;border-radius:50%;background:{colors.get(l, "#6b7280")};'
        f'color:white;font-weight:700;font-size:14px;margin-right:8px;">{l}</span>'
        for l in letters
    )
    st.markdown(f'<div style="margin:8px 0;">{spans}</div>', unsafe_allow_html=True)
