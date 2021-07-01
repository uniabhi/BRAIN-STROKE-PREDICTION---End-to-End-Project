import flask
import joblib
import os
import numpy as np
import pickle

app = flask.Flask(__name__)


@app.route("/")
def index():
    return flask.render_template("home.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    gender = int(flask.request.form['gender'])
    age = int(flask.request.form['age'])
    hypertension = int(flask.request.form['hypertension'])
    heart_disease = int(flask.request.form['heart_disease'])
    ever_married = int(flask.request.form['ever_married'])
    work_type = int(flask.request.form['work_type'])
    Residence_type = int(flask.request.form['Residence_type'])
    avg_glucose_level = float(flask.request.form['avg_glucose_level'])
    bmi = float(flask.request.form['bmi'])
    smoking_status = int(flask.request.form['smoking_status'])

    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                  avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    scaler_path = os.path.join('C:\RESUME PROJECT FILE\Stroke-Risk-Prediction-using-Machine-Learning-master\Stroke-Risk-Prediction-using-Machine-Learning-master', 'models/scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    x = scaler.transform(x)

    model_path = os.path.join('C:\RESUME PROJECT FILE\Stroke-Risk-Prediction-using-Machine-Learning-master\Stroke-Risk-Prediction-using-Machine-Learning-master', 'models/dt.sav')
    dt = joblib.load(model_path)

    Y_pred = dt.predict(x)

    # for No Stroke Risk
    if Y_pred == 0:
        return flask.render_template('nostroke.html')
    else:
        return flask.render_template('stroke.html')


if __name__ == "__main__":
    app.run(debug=True, port=7384)
