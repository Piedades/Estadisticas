"""
Sincronización con GitHub: permite subir datos nuevos y guardar el diario
de apuestas directamente en el repositorio, sin salir de la app.

Requiere un secret GITHUB_TOKEN (Settings > Secrets en Streamlit Cloud):
un Personal Access Token de GitHub con permiso 'repo' (o 'contents:write'
si usas un token fine-grained), generado en:
https://github.com/settings/tokens
"""
import base64

import requests
import streamlit as st

REPO = "Piedades/Estadisticas"
API_BASE = f"https://api.github.com/repos/{REPO}/contents"


def _headers():
    token = st.secrets.get("GITHUB_TOKEN")
    if not token:
        return None
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def get_file(path: str):
    """Devuelve (contenido_texto, sha) o (None, None) si no existe o no hay token."""
    headers = _headers()
    if headers is None:
        return None, None
    resp = requests.get(f"{API_BASE}/{path}", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content, data["sha"]
    return None, None


def put_file(path: str, content: str, message: str):
    """Crea o actualiza un archivo en el repo. Devuelve (ok: bool, mensaje: str)."""
    headers = _headers()
    if headers is None:
        return False, (
            "No hay GITHUB_TOKEN configurado. Ve a Streamlit Cloud > tu app > "
            "Settings > Secrets y añade: GITHUB_TOKEN = \"tu-token\" "
            "(créalo en github.com/settings/tokens con permiso 'repo')."
        )
    _, sha = get_file(path)
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    resp = requests.put(f"{API_BASE}/{path}", headers=headers, json=payload)
    if resp.status_code in (200, 201):
        return True, "Guardado en GitHub correctamente."
    return False, f"Error {resp.status_code} al guardar: {resp.text[:200]}"
