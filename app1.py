import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
Model = pickle.load(open('Model_1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/', methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = Model.predict(final_features)

    if prediction == 1:
        return render_template('index.html', prediction_text='Person is satisfied.')
    else:
        return render_template('index.html', prediction_text='Person is neutral or dissatisfied.')



if __name__ == "__main__":
    app.run(debug=True)