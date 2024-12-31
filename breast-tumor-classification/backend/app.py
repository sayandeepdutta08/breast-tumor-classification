from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Parse input features from form data
            features = [
                float(request.form.get(f'feature{i}')) for i in range(1, 31)
            ]
            features = np.array(features).reshape(1, -1)

            # Make a prediction
            prediction = model.predict(features)[0]
            result = "Malignant" if prediction == 1 else "Benign"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
