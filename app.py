from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo y los transformadores
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        Origen = request.form['Origen']
        Destino = request.form['Destino']
        Distancia = float(request.form['Distancia'])
        Size = float(request.form['Size'])
        Peso = float(request.form['Peso'])
        Estacion = request.form['Estacion']
        Fragilidad = request.form['Fragilidad']
        Urgencia = request.form['Urgencia']
        
        # Crear un DataFrame con los datos
        data = pd.DataFrame([[Origen, Destino, Distancia, Size, Peso, Estacion, Fragilidad, Urgencia]],
                            columns=['Origen', 'Destino', 'Distancia', 'Size', 'Peso', 'Estacion', 'Fragilidad', 'Urgencia'])
        
        # Codificar y escalar los datos
        data[['Origen', 'Destino', 'Estacion', 'Fragilidad', 'Urgencia']] = encoder.transform(data[['Origen', 'Destino', 'Estacion', 'Fragilidad', 'Urgencia']])
        data_scaled = scaler.transform(data)
        
        # Verificar el número de características
        if data_scaled.shape[1] != len(model.coef_):
            raise ValueError(f"El número de características después de la transformación ({data_scaled.shape[1]}) no coincide con el número de características que el modelo espera ({len(model.coef_)}).")
        
        # Hacer la predicción
        prediction = model.predict(data_scaled)
        
        # Devolver la predicción como JSON
        return jsonify({'prediction': float(prediction[0])})
    
    except Exception as e:
        # Manejo de errores
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Cargar los modelos y transformadores
# scaler = joblib.load('models/modelo_standarScaler.pkl')
# pca = joblib.load('models/modelo_PCA.pkl')
# model = joblib.load('models/modelo_regresión.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form.to_dict()
#     data = [float(data['Distacia']), int(data['Cant_Casetas']), 
#             float(data['Gasto_Casetas']), float(data['Consumo_gas']), 
#             float(data['precio_gas']), float(data['gasto_total_gas'])]
    
#     # Convertir a un array numpy y transformar los datos
#     data = np.array(data).reshape(1, -1)
#     data_scaled = scaler.transform(data)
#     data_pca = pca.transform(data_scaled)
    
#     # Hacer la predicción
#     prediction = model.predict(data_pca)

#     return jsonify({'predicted_price': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)
