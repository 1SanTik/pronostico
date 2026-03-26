from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


# ─────────────────────────────────────────────
# MODELO 1 – PROMEDIO MÓVIL
# ─────────────────────────────────────────────

def modelo_promedio_movil(serie, N, pasos):
    """model → fit (rolling) → predict"""
    # model
    ventana = N
    # fit — promedio movil sobre historico
    rolling = serie.rolling(window=ventana).mean()
    pronostico_hist = rolling.shift(1)

    # predict — ventana se actualiza recursivamente en cada paso
    # CORRECCIÓN: antes se repetía el mismo valor; ahora cada nuevo valor
    # proyectado entra a la ventana para calcular el siguiente
    valores = serie.tolist()
    proyecciones = []
    for _ in range(pasos):
        siguiente = round(sum(valores[-ventana:]) / ventana, 2)
        proyecciones.append(siguiente)
        valores.append(siguiente)

    error_abs = (pronostico_hist - serie).abs()
    mape = round((error_abs / serie.replace(0, np.nan)).mean() * 100, 2)
    rmse = round(((pronostico_hist - serie) ** 2).mean() ** 0.5, 2)

    return {
        'proyecciones': proyecciones,
        'mape': mape if not np.isnan(mape) else 0,
        'rmse': rmse if not np.isnan(rmse) else 0,
    }


# ─────────────────────────────────────────────
# MODELO 2 – SUAVIZACIÓN EXPONENCIAL (Holt)
# ─────────────────────────────────────────────

def modelo_ses(serie, pasos):
    """model → fit → predict  (Holt con tendencia — statsmodels)"""
    # CORRECCIÓN: se reemplazó SimpleExpSmoothing por Holt para capturar
    # tendencia y generar proyecciones con pendiente, no línea plana.
    # model
    model = Holt(serie.values, initialization_method='estimated')
    # fit
    fit = model.fit(optimized=True)
    # predict
    forecast = fit.forecast(pasos)
    proyecciones = [round(float(v), 2) for v in forecast]

    fitted = fit.fittedvalues
    errores = serie.values - fitted
    mape = round(float(np.mean(np.abs(errores / serie.replace(0, np.nan).values))) * 100, 2)
    rmse = round(float(np.sqrt(np.mean(errores ** 2))), 2)

    return {
        'proyecciones': proyecciones,
        'mape': mape if not np.isnan(mape) else 0,
        'rmse': rmse if not np.isnan(rmse) else 0,
        'alpha': round(float(fit.params['smoothing_level']), 4),
    }


# ─────────────────────────────────────────────
# MODELO 3 – PROPHET (Meta)
# ─────────────────────────────────────────────

def modelo_prophet(serie, pasos, frecuencia):
    """model → fit → predict  (Prophet)"""
    freq_map = {
        'MS': 'MS',
        'W':  'W',
        'D':  'D',
        'QS': 'QS',
        'AS': 'AS',
    }
    freq = freq_map.get(frecuencia, 'MS')

    fechas = pd.date_range(start='2000-01-01', periods=len(serie), freq=freq)
    df_prophet = pd.DataFrame({'ds': fechas, 'y': serie.values})

    # model
    # CORRECCIÓN: se activa seasonality_mode='auto' y se deja que Prophet
    # detecte tendencia y estacionalidad; antes estaban todas desactivadas
    # lo que producía proyecciones planas.
    model = Prophet(
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto',
        uncertainty_samples=0
    )
    # fit
    model.fit(df_prophet)
    # predict
    future = model.make_future_dataframe(periods=pasos, freq=freq)
    forecast = model.predict(future)

    proyecciones = [round(float(v), 2)
                    for v in forecast['yhat'].tail(pasos).values]

    fitted = forecast['yhat'].head(len(serie)).values
    errores = serie.values - fitted
    mape = round(float(np.mean(np.abs(errores / serie.replace(0, np.nan).values))) * 100, 2)
    rmse = round(float(np.sqrt(np.mean(errores ** 2))), 2)

    return {
        'proyecciones': proyecciones,
        'mape': mape if not np.isnan(mape) else 0,
        'rmse': rmse if not np.isnan(rmse) else 0,
    }


# ─────────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ─────────────────────────────────────────────

def calcular_metricas(df, N, metodo, pasos, frecuencia):
    resultados = {}

    for columna in df.columns:
        serie = pd.to_numeric(df[columna], errors='coerce').dropna()
        if len(serie) < max(N, 3):
            continue

        etiquetas_hist = list(range(1, len(serie) + 1))
        etiquetas_fut  = list(range(len(serie) + 1, len(serie) + pasos + 1))

        # Ejecutar modelo seleccionado
        if metodo == 'promedio_movil':
            res = modelo_promedio_movil(serie, N, pasos)
        elif metodo == 'ses':
            res = modelo_ses(serie, pasos)
        elif metodo == 'prophet':
            res = modelo_prophet(serie, pasos, frecuencia)
        else:
            res = modelo_promedio_movil(serie, N, pasos)

        # Ejecutar los 3 siempre para la tabla comparativa
        comp_pm  = modelo_promedio_movil(serie, N, pasos)
        comp_ses = modelo_ses(serie, pasos)
        try:
            comp_pro = modelo_prophet(serie, pasos, frecuencia)
        except Exception:
            comp_pro = {'mape': 0, 'rmse': 0, 'proyecciones': [0]}

        resultados[columna] = {
            'historico':       serie.tolist(),
            'etiquetas':       etiquetas_hist,
            'proyecciones':    res['proyecciones'],
            'etiquetas_fut':   etiquetas_fut,
            'mape':  res['mape'],
            'rmse':  res['rmse'],
            'alpha': res.get('alpha', '-'),
            'comp': {
                'Promedio Movil':          {'mape': comp_pm['mape'],  'rmse': comp_pm['rmse']},
                'Suavizacion Exponencial': {'mape': comp_ses['mape'], 'rmse': comp_ses['rmse']},
                'Prophet':                 {'mape': comp_pro['mape'], 'rmse': comp_pro['rmse']},
            }
        }

    return resultados


# ─────────────────────────────────────────────
# RUTAS FLASK
# ─────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    datos_web  = None
    n_ventana  = 3
    metodo     = 'promedio_movil'
    pasos      = 6
    frecuencia = 'MS'

    if request.method == 'POST':
        n_ventana  = int(request.form.get('n_ventana', 3))
        metodo     = request.form.get('metodo', 'promedio_movil')
        pasos      = int(request.form.get('pasos', 6))
        frecuencia = request.form.get('frecuencia', 'MS')
        archivo    = request.files.get('archivo_csv')

        if archivo and archivo.filename != '':
            try:
                df = pd.read_csv(archivo, sep=';')
                datos_web = calcular_metricas(df, n_ventana, metodo, pasos, frecuencia)
            except Exception as e:
                print(f"Error: {e}")
                return f"Hubo un problema al procesar el archivo: {e}"

    return render_template('index.html',
                           datos=datos_web,
                           n=n_ventana,
                           metodo=metodo,
                           pasos=pasos,
                           frecuencia=frecuencia)


if __name__ == '__main__':
    app.run(debug=True)