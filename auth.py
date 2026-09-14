"""
Autenticación con usuarios individuales (no una contraseña compartida).

Los usuarios se guardan en users.json dentro del repositorio de GitHub,
con la contraseña siempre en forma de hash + sal (nunca en texto plano).

Para crear el primer usuario, o cualquier usuario nuevo, hace falta un
código de invitación (secret INVITE_CODE en Streamlit Cloud) — así
cualquiera con el enlace de la app no puede crearse una cuenta sin más.
"""
import hashlib
import json
import secrets as secrets_module

import streamlit as st

from github_sync import get_file, put_file

USERS_PATH = "users.json"
CONFIG_PATH = "config.json"


def _hash_password(password: str, salt: str = None) -> tuple:
    salt = salt or secrets_module.token_hex(16)
    h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return h, salt


def load_users() -> dict:
    content, _ = get_file(USERS_PATH)
    if not content:
        return {}
    return json.loads(content)


def save_users(users: dict):
    return put_file(USERS_PATH, json.dumps(users, indent=2), "Actualizar usuarios")


def load_config() -> dict:
    content, _ = get_file(CONFIG_PATH)
    if not content:
        return {}
    return json.loads(content)


def save_config(config: dict):
    return put_file(CONFIG_PATH, json.dumps(config, indent=2), "Actualizar configuración")


def delete_user(username: str):
    """Devuelve (ok: bool, mensaje: str)."""
    username = username.strip().lower()
    users = load_users()
    if username not in users:
        return False, "Ese usuario no existe."
    del users[username]
    return save_users(users)


def update_invite_code(new_code: str):
    """Cambia el código de invitación desde la propia app (se guarda en
    config.json, con prioridad sobre el secret INVITE_CODE)."""
    new_code = new_code.strip()
    if not new_code:
        return False, "El código no puede estar vacío."
    config = load_config()
    config["invite_code"] = new_code
    return save_config(config)


def is_admin(username: str) -> bool:
    admins = st.secrets.get("ADMIN_USERS", "")
    admin_list = [a.strip().lower() for a in admins.split(",") if a.strip()]
    return username.strip().lower() in admin_list


def _check_invite_code(code: str) -> bool:
    config = load_config()
    real_code = config.get("invite_code") or st.secrets.get("INVITE_CODE")
    if not real_code:
        return False
    return code == real_code


def register_user(username: str, password: str, invite_code: str):
    """Devuelve (ok: bool, mensaje: str)."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Usuario y contraseña no pueden estar vacíos."
    if len(password) < 6:
        return False, "La contraseña debe tener al menos 6 caracteres."
    if not _check_invite_code(invite_code):
        return False, "Código de invitación incorrecto."

    users = load_users()
    if username in users:
        return False, "Ese nombre de usuario ya existe."

    h, salt = _hash_password(password)
    users[username] = {"hash": h, "salt": salt}
    ok, msg = save_users(users)
    if not ok:
        return False, msg
    return True, "Cuenta creada. Ya puedes iniciar sesión."


def authenticate(username: str, password: str) -> bool:
    username = username.strip().lower()
    users = load_users()
    if username not in users:
        return False
    entry = users[username]
    h, _ = _hash_password(password, entry["salt"])
    return h == entry["hash"]
