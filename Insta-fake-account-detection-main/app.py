
# coding: utf-8

# In[1]:




import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/insta', methods=['POST', 'GET'])
def rinsta():
	return render_template('resulti.html')


@app.route('/resulti.html', methods=['POST', 'GET'])
def insta():
     float_features = [float(x) for x in request.form.values()]
     final_features = [np.array(float_features)]
     prediction = model.predict(final_features)
     if prediction == 0:
          pred ='REAL ACCOUNT'
     elif prediction == 1:
          pred = 'FAKE ACCOUNT'
     output = pred
     return render_template('resulti.html', prediction_text = 'This is {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)


