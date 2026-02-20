from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    x = float(data['value'])
    return jsonify({"result": x*2})

app.run()
