from flask import Flask, jsonify
import pandas as pd

# Definir el DataFrame df_interfaz
df_interfaz = pd.DataFrame({
    'Medicamento': ['Med1', 'Med2', 'Med3'],
    'Fecha': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Stock Inicial': [100, 150, 200],
    'Stock Final': [90, 140, 190]
})

# Definir el DataFrame df_prediccion
df_prediccion = pd.DataFrame({
    'Medicamento': ['Med1', 'Med2', 'Med3'],
    'Fecha': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Stock Inicial': [100, 150, 200],
    'Stock Final': [90, 140, 190]
})

app = Flask(__name__)

# Ruta para obtener el total de medicamentos
@app.route('/api/total_medicamentos', methods=['GET'])
def total_medicamentos():
    total = df_interfaz['Medicamento'].nunique()
    return jsonify({'total_medicamentos': total})

# Ruta para obtener el nivel promedio de stock
@app.route('/api/nivel_promedio_stock', methods=['GET'])
def nivel_promedio_stock():
    promedio = df_interfaz['Stock Final'].mean()
    return jsonify({'nivel_promedio_stock': promedio})

# Ruta para obtener los medicamentos con stock bajo
@app.route('/api/medicamentos_stock_bajo', methods=['GET'])
def medicamentos_stock_bajo():
    medicamentos = df_interfaz.loc[df_interfaz['Stock Final'] < 10, 'Medicamento'].tolist()
    return jsonify({'medicamentos_stock_bajo': medicamentos})

# Ruta para obtener el mapa de calor de riesgo
@app.route('/api/mapa_calor_riesgo', methods=['GET'])
def mapa_calor_riesgo():
    # Aquí se debe generar el mapa de calor usando df_riesgo
    # y convertirlo a un formato que se pueda enviar en la respuesta JSON
    # Por ejemplo, se puede usar la librería matplotlib para generar una imagen
    # y luego codificarla en base64 para enviarla en la respuesta
    # ...
    return jsonify({'mapa_calor_riesgo': 'imagen_codificada_en_base64'})

# Ruta para obtener los datos para el gráfico
@app.route('/api/datos_grafico', methods=['GET'])
def datos_grafico():
    # Convertir los datos del DataFrame a un formato adecuado para JSON
    data = df_prediccion[['Medicamento', 'Fecha', 'Stock Inicial', 'Stock Final']].to_dict(orient='records')
    return jsonify({'datos_grafico': data})

# Ruta para obtener los datos para la interfaz
@app.route('/api/datos_interfaz', methods=['GET'])
def datos_interfaz():
    # Convertir los datos del DataFrame a un formato adecuado para JSON
    data = df_interfaz.to_dict(orient='records')
    return jsonify({'datos_interfaz': data})

if __name__ == '__main__':
    app.run(debug=True)