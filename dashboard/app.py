import os, io, base64
import joblib
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# Configuración básica
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "RandomForest.pkl")
TARGET_CANDIDATES = ["Churn", "churn", "CHURN"]
DEFAULT_THR = 0.5

model = joblib.load(MODEL_PATH)

# Utilidades
def read_csv(contents):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))

def find_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    return None

def prepare(df, expected=None, target=None):
    if target and target in df.columns:
        df = df.drop(columns=[target])
    cat = df.select_dtypes(include=["object"]).columns.tolist()
    if cat:
        df = pd.get_dummies(df, columns=cat, drop_first=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0)
    if expected is not None:
        cols = list(expected)
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        df = df[cols]
    else:
        df = df.select_dtypes(include=["number"]).copy()
    return df

def kpi_box(title, value):
    return html.Div(
        [html.Div(title, style={"fontSize": "12px", "color": "#555"}),
         html.Div(value, style={"fontSize": "22px", "fontWeight": "600"})],
        style={"border": "1px solid #ddd", "borderRadius": "8px",
               "padding": "10px 14px", "minWidth": "180px"}
    )

# App
app = Dash(__name__)
app.title = "Dashboard de Predicción"

app.layout = html.Div([
    html.H2("Modelo de Riesgo de abandono de Clientes",
            style={"textAlign": "center", "marginBottom": "10px"}),

    dcc.Upload(
        id="file", multiple=False,
        children=html.Div(["Arrastra o selecciona un ", html.B(".csv")]),
        style={"width": "100%", "height": "60px", "lineHeight": "60px",
               "border": "1px dashed #999", "borderRadius": "6px",
               "textAlign": "center", "margin": "10px 0"}
    ),

    html.Div([
        html.Div([html.Label("Umbral"),
                  dcc.Slider(id="thr", min=0.1, max=0.9, step=0.05, value=DEFAULT_THR,
                             marks={i/10: f"{i/10:.1f}" for i in range(1, 10)})],
                 style={"flex": "2"}),
        html.Div(id="thr_lbl", style={"alignSelf": "flex-end", "marginLeft": "12px"})
    ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),

    # Filtros
    html.Div([
        html.Div([html.Label("Filtro país"),   dcc.Dropdown(id="f_country", placeholder="Todos")]),
        html.Div([html.Label("Filtro género"), dcc.Dropdown(id="f_gender",  placeholder="Todos")]),
        html.Div([html.Label("Filtro edad"),
                  dcc.RangeSlider(id="f_age", min=0, max=100, step=1,
                                  value=[0, 100], allowCross=False)]),
        html.Div([html.Label("Filtro ingresos"),
                  dcc.RangeSlider(id="f_income", min=0, max=300000, step=1000,
                                  value=[0, 300000], allowCross=False)]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)",
              "gap": "10px", "margin": "10px 0"}),

    html.Div(id="kpis", style={"display": "flex", "gap": "12px",
                               "flexWrap": "wrap", "margin": "8px 0"}),

    # Cuadrícula 2x2 (maqueta)
    html.Div([
        html.Div([dcc.Graph(id="g_dist")], style={"minWidth": "0"}),
        html.Div([dcc.Graph(id="g_map")], style={"minWidth": "0"}),
        html.Div([dcc.Graph(id="g_imp")], style={"minWidth": "0"}),
        html.Div([
            html.H4("Top clientes de riesgo", style={"marginTop": "0"}),
            dash_table.DataTable(
                id="tbl_top", page_size=10,
                style_table={"overflowX": "auto", "maxHeight": "420px", "overflowY": "auto"}
            )
        ], style={"minWidth": "0"})
    ], style={"display": "grid",
              "gridTemplateColumns": "1fr 1fr",
              "gap": "16px", "alignItems": "start", "marginTop": "16px"})
])

@app.callback(Output("thr_lbl", "children"), Input("thr", "value"))
def show_thr(v):
    return f"Umbral actual: {v:.2f}"

@app.callback(
    Output("g_dist", "figure"), Output("g_map", "figure"), Output("g_imp", "figure"),
    Output("tbl_top", "data"), Output("tbl_top", "columns"),
    Output("f_country", "options"), Output("f_gender", "options"),
    Output("f_age", "min"), Output("f_age", "max"), Output("f_age", "value"),
    Output("f_income", "min"), Output("f_income", "max"), Output("f_income", "value"),
    Output("kpis", "children"),
    Input("file", "contents"), Input("thr", "value"),
    Input("f_country", "value"), Input("f_gender", "value"),
    Input("f_age", "value"), Input("f_income", "value"),
    State("file", "filename"),
    prevent_initial_call=True
)
def run(contents, thr, v_country, v_gender, v_age, v_income, filename):
    empty_fig = go.Figure()

    if contents is None:
        return (empty_fig, empty_fig, empty_fig, [], [],
                [], [], 0, 100, [0, 100], 0, 300000, [0, 300000], [])

    try:
        df_raw = read_csv(contents)
    except Exception:
        return (empty_fig, empty_fig, empty_fig, [], [],
                [], [], 0, 100, [0, 100], 0, 300000, [0, 300000], [])

    target = find_target(df_raw)
    expected = getattr(model, "feature_names_in_", None)
    df_model = prepare(df_raw.copy(), expected=expected, target=target)

    has_country = "country" in df_raw.columns
    has_gender  = "gender"  in df_raw.columns
    has_age     = "age"     in df_raw.columns
    has_income  = "estimated_salary" in df_raw.columns

    opts_country = [{"label": x, "value": x} for x in sorted(df_raw["country"].dropna().unique())] if has_country else []
    opts_gender  = [{"label": x, "value": x} for x in sorted(df_raw["gender"].dropna().unique())]  if has_gender  else []

    age_min, age_max = (int(df_raw["age"].min()), int(df_raw["age"].max())) if has_age else (0, 100)
    inc_min, inc_max = (float(df_raw["estimated_salary"].min()), float(df_raw["estimated_salary"].max())) if has_income else (0.0, 300000.0)

    age_val = v_age if (v_age and has_age) else [age_min, age_max]
    inc_val = v_income if (v_income and has_income) else [inc_min, inc_max]

    mask = pd.Series(True, index=df_raw.index)
    if has_country and v_country: mask &= (df_raw["country"] == v_country)
    if has_gender  and v_gender:  mask &= (df_raw["gender"]  == v_gender)
    if has_age:    mask &= df_raw["age"].between(age_val[0], age_val[1])
    if has_income: mask &= df_raw["estimated_salary"].between(inc_val[0], inc_val[1])

    df_raw_f   = df_raw.loc[mask]
    df_model_f = df_model.loc[mask]

    if len(df_model_f) == 0:
        return (empty_fig, empty_fig, empty_fig, [], [],
                opts_country, opts_gender, age_min, age_max, age_val,
                inc_min, inc_max, inc_val, [])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_model_f)[:, 1]
    else:
        raw = model.predict(df_model_f)
        probs = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    df_pred = df_model_f.copy()
    df_pred["probabilidad"] = probs
    df_pred["prediccion"] = (df_pred["probabilidad"] >= thr).astype(int)
    df_pred["accion"] = np.where(df_pred["probabilidad"] >= thr,
                                 "Prioridad ALTA de retención", "Seguimiento normal")

    # KPIs
    kpis = []
    kpis.append(kpi_box("% Riesgo ALTO", f"{(df_pred['probabilidad'] >= thr).mean() * 100:.1f}%"))
    if target:
        y_true = df_raw_f[target]
        if y_true.dtype == "O":
            y_true = y_true.astype(str).str.lower().map(
                {"yes": 1, "si": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
            ).fillna(0).astype(int)
        else:
            y_true = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)
        from sklearn.metrics import accuracy_score, f1_score
        kpis += [
            kpi_box("KPI Churn rate", f"{y_true.mean() * 100:.1f}%"),
            kpi_box("KPI Accuracy", f"{accuracy_score(y_true, df_pred['prediccion']) * 100:.1f}%"),
            kpi_box("KPI F1-score", f"{f1_score(y_true, df_pred['prediccion']):.2f}")
        ]
    else:
        kpis += [kpi_box("KPI Churn rate", "N/A"),
                 kpi_box("KPI Accuracy", "N/A"),
                 kpi_box("KPI F1-score", "N/A")]

    # Gráficos
    fig_dist = px.histogram(df_pred, x="probabilidad", nbins=20, title="Distribución de Riesgo")

    if has_country and len(df_raw_f):
        tmp = pd.DataFrame({"country": df_raw_f["country"], "prob": df_pred["probabilidad"]})
        geo = tmp.groupby("country", as_index=False)["prob"].mean()
        fig_map = px.choropleth(geo, locations="country", locationmode="country names",
                                color="prob", color_continuous_scale="Blues",
                                title="Mapa de riesgo promedio por país")
    else:
        fig_map = empty_fig

    if hasattr(model, "feature_importances_"):
        feats_attr = getattr(model, "feature_names_in_", None)
        feats = [f"f{i}" for i in range(len(model.feature_importances_))] if feats_attr is None else list(feats_attr)
        fi = (pd.DataFrame({"feature": feats, "importance": model.feature_importances_})
                .sort_values("importance", ascending=False).head(12))
        fig_imp = px.bar(fi, x="importance", y="feature", orientation="h",
                         title="Variables más importantes (top 12)")
    else:
        fig_imp = empty_fig

    # Top clientes
    df_top = pd.DataFrame({"probabilidad": df_pred["probabilidad"], "prediccion": df_pred["prediccion"]})
    for cand in ["customer_id", "CustomerID", "id", "ID"]:
        if cand in df_raw_f.columns:
            df_top[cand] = df_raw_f[cand].values
            break
    df_top = df_top.sort_values("probabilidad", ascending=False).head(10)
    top_cols = [{"name": c, "id": c} for c in df_top.columns]
    top_data = df_top.to_dict("records")

    return (fig_dist, fig_map, fig_imp,
            top_data, top_cols,
            opts_country, opts_gender,
            age_min, age_max, age_val,
            inc_min, inc_max, inc_val,
            kpis)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)