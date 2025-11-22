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

## Despliegue en AWS EC2

Para ejecutar este tablero en una instancia EC2 (como lo requiere la entrega final del curso), siga estos pasos:

1. Conectarse a la instancia: ssh ubuntu@<IP_PUBLICA_EC2>
2. Instalar dependencias básicas:
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
3. Clonar el repositorio:
git clone https://github.com/FelipeRubiano/MIAD_DSA_PROYECTO.git
cd MIAD_DSA_PROYECTO
4. Crear y activar entorno virtual:
python3 -m venv .venv
source .venv/bin/activate
5. Instalar dependencias:
pip install -r requirements.txt
6. Ejecutar el tablero:
python dashboard/app.py

El tablero quedará disponible en:

http://<IP_PUBLICA_EC2>:8050

Recuerde abrir el puerto 8050 en el Security Group de la instancia.

