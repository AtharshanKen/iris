import numpy as np
import pickle
from flask import Flask, request, render_template, current_app

app = Flask(__name__, template_folder="templates")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier.pkl")
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                classifier = pickle.load(f)
        except FileNotFoundError:
            # helpful error in logs if model not present
            current_app.logger.error(f"Model file not found at {MODEL_PATH}")
            raise
        except Exception as e:
            current_app.logger.error(f"Error loading model: {e}")
            raise
    return classifier

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form input values to floats. This will raise ValueError for bad input.
        raw = [v for v in request.form.values()]
        features = [float(x) for x in raw]
    except ValueError:
        # bad user input
        return render_template('index.html', prediction_text="Invalid input: please submit numeric values.")

    clf = get_classifier()  # lazy-load model (raises if missing)

    # scikit-learn expects 2D array: shape (1, n_features)
    final_features = np.array([features])
    try:
        prediction = clf.predict(final_features)
        output = prediction[0]
    except Exception as e:
        current_app.logger.error(f"Prediction error: {e}")
        return render_template('index.html', prediction_text=f"Prediction error: {e}")

    return render_template('index.html', prediction_text=f'The flower belongs to species {output}')

# Only used when running locally (gunicorn will import app and run it)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
