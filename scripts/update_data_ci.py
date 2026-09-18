"""
Script independiente usado por el GitHub Action programado
(.github/workflows/update-data.yml) para descargar los datos más recientes
de football-data.co.uk y sobreescribir los CSV de la temporada en curso.

No usa github_sync.py a propósito: ese módulo depende de st.secrets, que
solo existe dentro de una app de Streamlit. Aquí, en un runner de GitHub
Actions, simplemente sobreescribimos los archivos del checkout local y
dejamos que el workflow los commitee con git.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_update import CURRENT_SEASON_FILE, fetch_league_csv  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    changed = []
    for league_code, filename in CURRENT_SEASON_FILE.items():
        try:
            csv_text = fetch_league_csv(league_code)
        except Exception as e:
            print(f"[{league_code}] error al descargar: {e}")
            continue

        if len(csv_text) < 100:
            print(f"[{league_code}] descarga vacía o incompleta, se ignora.")
            continue

        path = DATA_DIR / filename
        old_text = path.read_text(encoding="utf-8-sig") if path.exists() else None
        if old_text == csv_text:
            print(f"[{league_code}] sin cambios.")
            continue

        path.write_text(csv_text, encoding="utf-8")
        changed.append(filename)
        print(f"[{league_code}] actualizado: {filename}")

    if changed:
        print("Archivos modificados:", ", ".join(changed))
    else:
        print("No hubo cambios en ninguna liga.")


if __name__ == "__main__":
    main()
