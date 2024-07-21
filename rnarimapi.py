import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras.backend as K
from flask import Flask, jsonify
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, Dropout

app = Flask(__name__)

# Leer el archivo CSV en un DataFrame
df = pd.read_csv('datos_entrenamiento_desabastecimiento.csv')

# Convertir 'Fecha' a datetime
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Calcular consumo diario por medicamento
df_consumo_diario = df.groupby(['Fecha', 'Medicamento'])['Consumo Histórico'].sum().reset_index()

# Crear columna 'Desabastecimiento'
df['Desabastecimiento'] = (df['Inventario Actual'] < df['Consumo Histórico']).astype(int)

# Codificar variables categóricas
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cols = encoder.fit_transform(df[['Medicamento', 'Nivel de Ingresos (Bajo/Medio/Alto)']])

# Obtener los nombres de las columnas originales utilizadas para la codificación
feature_names_in = encoder.feature_names_in_

# Obtener los nombres de las columnas codificadas
encoded_column_names = encoder.get_feature_names_out(feature_names_in)

encoded_df = pd.DataFrame(encoded_cols, columns=encoded_column_names)
df = pd.concat([df, encoded_df], axis=1)

# Normalizar las características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[encoded_column_names])
df[encoded_column_names] = scaled_features

def fit_model(train, X_train):
    # Crear el modelo
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenar el modelo
    y_train = train['Desabastecimiento']
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    return model

@app.route('/total_medicamentos', methods=['GET'])
def total_medicamentos():
    total_medicamentos = len(df['Medicamento'].unique())
    nombres_medicamentos = df['Medicamento'].unique().tolist()
    return jsonify({
        'total_medicamentos': total_medicamentos,
        'nombres_medicamentos': nombres_medicamentos
    })

@app.route('/medicamentos_stock_bajo', methods=['GET'])
def medicamentos_stock_bajo():
    umbral_stock_bajo = 10
    medicamentos_stock_bajo = df[df['Inventario Actual'] < umbral_stock_bajo]['Medicamento'].unique().tolist()
    return jsonify({
        'medicamentos_stock_bajo': medicamentos_stock_bajo
    })

@app.route('/nivel_promedio_stock', methods=['GET'])
def nivel_promedio_stock():
    nivel_promedio_stock = df['Inventario Actual'].mean()
    return jsonify({
        'nivel_promedio_stock': float(nivel_promedio_stock)  # Convertir a float
    })

@app.route('/prediccion_stock', methods=['GET'])
def prediccion_stock():
    # Definir el umbral para desabastecimiento
    umbral_desabastecimiento = 10

    # Agregar la columna 'Desabastecimiento' al DataFrame
    df['Desabastecimiento'] = (df['Inventario Actual'] < umbral_desabastecimiento).astype(int)

    predicciones = []
    metrics = []
    for medicamento in df['Medicamento'].unique():
        df_medicamento = df_consumo_diario[df_consumo_diario['Medicamento'] == medicamento].set_index('Fecha')
        train_size = int(len(df_medicamento) * 0.8)
        train, test = df_medicamento.iloc[:train_size], df_medicamento.iloc[train_size:]

        model_arima = ARIMA(train['Consumo Histórico'], order=(1, 0, 0)).fit()
        residuals = model_arima.resid

        train_encoded = df[df['Medicamento'] == medicamento].iloc[:train_size][encoded_df.columns]
        X_train = pd.concat([residuals, train_encoded], axis=1).dropna()
        
        X_test = pd.concat([test['Consumo Histórico'], df[df['Medicamento'] == medicamento].iloc[train_size:][encoded_df.columns]], axis=1).dropna()
        
        K.clear_session()
        model = fit_model(train, X_train)

        if model is None:
            raise ValueError("El modelo no se ha entrenado correctamente")

        y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()  # Aplanar para evitar problemas de formato

        # Calcular métricas
        y_train = df[df['Medicamento'] == medicamento].iloc[:train_size]['Desabastecimiento']
        y_train = y_train[y_train.index.isin(X_train.index)]
        y_test = df[df['Medicamento'] == medicamento].iloc[train_size:]['Desabastecimiento']
        y_test = y_test[y_test.index.isin(X_test.index)]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics.append({
            'medicamento': medicamento,
            'accuracy': float(accuracy),  # Convertir a float
            'precision': float(precision),  # Convertir a float
            'recall': float(recall),  # Convertir a float
            'f1_score': float(f1),  # Convertir a float
            'roc_auc': float(roc_auc),  # Convertir a float
            'confusion_matrix': conf_matrix.tolist()
        })

        fecha_actual = datetime.now().strftime('%Y-%m-%d')
        fecha_a_predecir = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        for i in range(len(y_pred)):
            if i < len(test):
                predicciones.append({
                    'medicamento': medicamento,
                    'fecha_prediccion': fecha_a_predecir,
                    'nivel_riesgo': 'Alto' if y_pred[i] == 1 else 'Bajo',
                    'stock_actual': int(test['Consumo Histórico'].iloc[i]),  # Convertir a int
                    'stock_predicho': int(y_pred[i])  # Convertir a int
                })
            else:
                predicciones.append({
                    'medicamento': medicamento,
                    'fecha_prediccion': fecha_a_predecir,
                    'nivel_riesgo': 'Alto' if y_pred[i] == 1 else 'Bajo',
                    'stock_actual': None,  # O algún valor por defecto
                    'stock_predicho': int(y_pred[i])  # Convertir a int
                })

    return jsonify({'predicciones': predicciones, 'metrics': metrics})


@app.route('/datos_grafico_series', methods=['GET'])
def datos_grafico_series():
    data_graph = []
    for medicamento in df['Medicamento'].unique():
        df_medicamento = df_consumo_diario[df_consumo_diario['Medicamento'] == medicamento].set_index('Fecha')
        train_size = int(len(df_medicamento) * 0.8)
        train, test = df_medicamento.iloc[:train_size], df_medicamento.iloc[train_size:]

        model_arima = ARIMA(train['Consumo Histórico'], order=(1, 0, 0))
        model_fit = model_arima.fit()
        residuals = model_fit.resid

        X_train = pd.concat([residuals, df[df['Medicamento'] == medicamento].iloc[:train_size][encoded_df.columns]], axis=1)
        X_train = X_train.dropna(subset=encoded_df.columns)
        y_train = df[df['Medicamento'] == medicamento].iloc[:train_size]['Desabastecimiento']
        y_train = y_train[y_train.index.isin(X_train.index)]

        X_test = pd.concat([test['Consumo Histórico'], df[df['Medicamento'] == medicamento].iloc[train_size:][encoded_df.columns]], axis=1)
        X_test = X_test.dropna(subset=encoded_df.columns)
        y_test = df[df['Medicamento'] == medicamento].iloc[train_size:]['Desabastecimiento']
        y_test = y_test[y_test.index.isin(X_test.index)]

        K.clear_session()
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        y_pred = (model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        data_graph.append({
            'medicamento': medicamento,
            'fechas': df_medicamento.index.strftime('%Y-%m-%d').tolist(),
            'stock_actual': df_medicamento['Consumo Histórico'].tolist(),
            'stock_predicho': y_pred.tolist()
        })

    return jsonify(data_graph)

if __name__ == '__main__':
    app.run(debug=True)