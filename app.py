from flask import Flask, redirect, render_template, request
import pandas as pd

from titanic_model import deploy_model

app = Flask(__name__)


@app.route('/')
def main():
    return redirect('/index')


@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    df = process_inputs()
    proba = deploy_model(df)

    return "Probability of survival is: {0:.2f}%".format(100 * proba[0, 1])


def process_inputs():
    '''
    Process input data for the model training.
    '''

    float_keys = ('Age', 'Fare')
    int_keys = ('Pclass', 'SibSp', 'Parch')
    inputs = {}

    for key, val in request.form.items():
        if key in float_keys:
            inputs[key] = float(val)
        elif key in int_keys:
            inputs[key] = int(val)
        else:
            inputs[key] = val

    return pd.DataFrame(inputs, index=[0])

if __name__ == '__main__':
    app.run(debug = True)
