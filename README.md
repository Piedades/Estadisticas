# Panel de predicción — Big 5 Ligas Europeas

Dashboard local para consultar predicciones, ratings Elo y estadísticas
de las 5 grandes ligas, usando el modelo Poisson/Dixon-Coles del proyecto.

## Instalación (una sola vez)

Necesitas Python 3.9 o superior instalado. Luego, desde esta carpeta:

```bash
pip install -r requirements.txt
```

## Arrancar la app

```bash
streamlit run app.py
```

Se abrirá automáticamente en tu navegador en `http://localhost:8501`.
Para cerrarla, vuelve a la terminal y pulsa Ctrl+C.

## Qué incluye

- **Predecir partido**: elige local y visitante de una liga y obtén
  probabilidades 1X2, goles esperados, marcador más probable,
  over/under 2.5, y córners/tiros/tarjetas esperados.
- **Ranking Elo**: fuerza actual de cada equipo.
- **Ratings por equipo**: ataque/defensa en córners, tiros, tiros a
  puerta y tarjetas amarillas.
- **Estadísticas de la liga**: evolución de goles y resultados por
  temporada.

## Actualizar los datos

Los CSV están en la carpeta `data/`. Para añadir una temporada nueva,
simplemente copia el archivo (mismo formato de football-data.co.uk)
dentro de `data/` y vuelve a arrancar la app — no hace falta tocar
el código.

## Todo funciona en local

Esta app no se conecta a internet ni envía datos a ningún sitio.
Todo el cálculo (modelo, Elo, ratings) se hace en tu propio ordenador
con los CSV que ya tienes.
