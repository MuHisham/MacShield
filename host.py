from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load('svm_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Serves the empty HTML file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processeddata = [[data['slider1'], data['slider2'], data['slider3'], data['slider4'], data['slider5']]]
    prediction = model.predict(processeddata)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False)
