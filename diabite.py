from asyncio import events
from cmath import polar
from flask import Flask, render_template, request
import numpy as np
import joblib
import json
import flask
model =joblib.load('D:\\Test Api\\RF.mdl')

app = Flask(__name__,template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def man():
    return render_template('diabetes.html')


@app.route('/predict', methods=['POST'])
def home():
    data = {"success": False}
    
    Polydipsia = request.form['Polydipsia']
    Polyuria = request.form['Polyuria']
    Gender = request.form['Gender']
    Age = request.form['Age']
    Sudden_Weight_Loss = request.form['Sudden Weight Loss']
    Alopecea = request.form['Alopecea']
    
    ressdk = {"Polydipsia":Polydipsia,
               "Polyuria":Polyuria,
               "Gender":Gender,
               "Age":Age,
               "Sudden_Weight_Loss":Sudden_Weight_Loss,
               "Alopecea":Alopecea} 
    
    json_object = json.dumps(ressdk) 
    
    arr = np.array([[Polydipsia, Polyuria, Gender, Age, Sudden_Weight_Loss, Alopecea]])
    res = model.predict(arr)
   
    if res ==1:
        pred="Positive"
    else:
        pred = "Negative"

    ressdk["predict"]=pred

    data["predictions"] = []
    data["predictions"].append(pred)
    data["success"] = True
    return render_template('diabetes.html', pred=res )
    #return('diabetes.html',str(res[0]))
 

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)