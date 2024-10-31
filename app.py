from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregar o modelo
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Obter os dados da requisição
    data = request.get_json(force=True)

    # Verificar se 'userScore' está no JSON
    if 'userScore' not in data:
        return jsonify({'error': 'No features provided'}), 400

    # Fazer a previsão
    try:
        userScore = data['userScore']
        # As entradas são esperadas como scores (ex.: [25], [75], etc.)
        userScore_codificadas = [[int(item)] for item in userScore]

        # Fazer a previsão
        prediction = model.predict(userScore_codificadas)

        # Mapear as previsões para suas respectivas classes
        classe_mapping = {0: 'Ruim', 1: 'Médio', 2: 'Bom', 3: 'Ótimo'}
        classes_previstas = [classe_mapping[pred] for pred in prediction]

        # Retornar as previsões como JSON
        return jsonify({'predictions': classes_previstas})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)