# In the Flask app
from flask import Flask, request, jsonify, render_template
from model_utils import make_prediction, train_model, load_model

app = Flask(__name__)

train = train_model()
# Load the model at the start
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['longitude']),
            float(request.form['latitude']),
            float(request.form['median_income']),
            int(request.form['ocean_proximity_INLAND']),
            int(request.form['ocean_proximity_NEAR_OCEAN']),
            float(request.form['housing_median_age']),
            float(request.form['total_rooms']),
            float(request.form['total_bedrooms'])
        ]
        prediction = make_prediction(model, input_data)
        return jsonify(prediction=prediction)
    except KeyError as e:
        return jsonify(error=f'Missing key: {str(e)}'), 400
    except ValueError as e:
        return jsonify(error=f'Invalid input: {str(e)}'), 400
    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
