# Dashboard del Modelo de Riesgo de Abandono de Clientes

Este tablero fue desarrollado con Dash para la Entrega 2 del curso *Despliegue de Soluciones Analíticas*.

## Ejecución

1. Activa el entorno virtual:
   source .venv/bin/activate

2. Instala dependencias:
   pip install -r requirements.txt

3. Ejecuta el tablero:
   python dashboard/app.py

El tablero se abrirá en http://0.0.0.0:8050

## Descripción

El tablero permite:
- Cargar un archivo CSV con datos de clientes.
- Ajustar el umbral de clasificación.
- Filtrar por país, género, edad e ingresos.
- Visualizar KPIs (Churn rate, Accuracy, F1-score).
- Ver la distribución de riesgo, el mapa de riesgo promedio por país,
  las variables más importantes y el top de clientes con mayor probabilidad de abandono.

