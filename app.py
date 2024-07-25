from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar los modelos y transformadores
scaler = joblib.load('models/modelo_standarScaler.pkl')
pca = joblib.load('models/modelo_PCA.pkl')
model = joblib.load('models/modelo_regresión.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = [float(data['Distacia']), int(data['Cant_Casetas']), 
            float(data['Gasto_Casetas']), float(data['Consumo_gas']), 
            float(data['precio_gas']), float(data['gasto_total_gas'])]
    
    # Convertir a un array numpy y transformar los datos
    data = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data)
    data_pca = pca.transform(data_scaled)
    
    # Hacer la predicción
    prediction = model.predict(data_pca)

    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
