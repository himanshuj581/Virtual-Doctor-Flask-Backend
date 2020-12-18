
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
from flask_restful import reqparse
app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/heartDisease', methods=['POST'])
def heartDisease():
    lr = joblib.load("heartDiseaseModel.pkl")
    if lr:       
        try:
            json = request.get_json() 
            model_columns = joblib.load("heartDiseaseModelCols.pkl")
            m = pd.DataFrame(json,columns=model_columns)
            prediction = lr.predict(m)
            print("here:",prediction)        
            return jsonify({'prediction': str(prediction[0])})
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')

@app.route('/diabetes', methods=['POST'])
def diabetes():
    lr = joblib.load("diabetesModel.pkl")
    if lr:       
        try:
            json = request.get_json() 
            model_columns = joblib.load("diabetesModelCols.pkl")
            m = pd.DataFrame(json,columns=model_columns)
            prediction = lr.predict(m)
            print("here:",prediction)        
            return jsonify({'prediction': str(prediction[0])})
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')

@app.route('/predict', methods=['POST'])
def predict():
    lr = joblib.load("model.pkl")
    if lr:       
        try:
            json = request.get_json()  
            model_columns = joblib.load("model_cols.pkl")
            temp=list(json[0])
            vals=np.array(temp)
            temp=[temp]
            prediction = lr.predict(temp)
            print("here:",prediction)        
            return jsonify({'disease': str(prediction[0])})
        except:        
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('No model here to use')
        
if __name__ == '__main__':
    app.run(debug=True)

