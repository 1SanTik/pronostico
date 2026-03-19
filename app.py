from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

def calcular_metricas(df, N):
    resultados = {}
    for columna in df.columns:
        serie = pd.to_numeric(df[columna], errors='coerce').dropna()
        
        if len(serie) < N:
            continue
        pronostico_serie = serie.rolling(window=N).mean().shift(1)
        error_abs = (pronostico_serie - serie).abs()
        # MAPE (Error porcentual medio)
        mape = (error_abs / serie).mean() * 100
        # RMSE (Raíz del error cuadrático medio)
        rmse = ((pronostico_serie - serie)**2).mean()**0.5
        # El valor para el próximo mes
        futuro = serie.tail(N).mean()

        resultados[columna] = {
            'mape': round(mape, 2) if not pd.isna(mape) else 0,
            'rmse': round(rmse, 2) if not pd.isna(rmse) else 0,
            'futuro': round(futuro, 2),
            'historico': serie.tolist(),
            'etiquetas': list(range(1, len(serie) + 1))
        }
    return resultados

@app.route('/', methods=['GET', 'POST'])
def index():
    datos_web = None
    n_ventana = 3
    
    if request.method == 'POST':
        n_ventana = int(request.form.get('n_ventana', 3))
        archivo = request.files.get('archivo_csv')
        
        if archivo and archivo.filename != '':
            try:
                df = pd.read_csv(archivo, sep=';')
                datos_web = calcular_metricas(df, n_ventana)
            except Exception as e:
                print(f"Error: {e}")
                return "Hubo un problema con el formato del archivo CSV."
            
    return render_template('index.html', datos=datos_web, n=n_ventana)

if __name__ == '__main__':
    app.run(debug=True)