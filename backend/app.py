# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_disease_from_text
import json
from Treatment import diseaseDetail

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('symptoms', '')
    if not isinstance(text, str) or text.strip() == '':
        return jsonify({'error': 'Please provide symptoms as a non-empty string under "symptoms"'}), 400
    try:
        res = predict_disease_from_text(text, top_k=10)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/treatment', methods=['GET'])
def treatment():
    disease = request.args.get('disease')
    if not disease:
        return jsonify({'error': 'provide disease query param, e.g. ?disease=Flu'}), 400
    try:
        detail = diseaseDetail(disease)
        return jsonify({'disease': disease, 'treatment': detail})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)