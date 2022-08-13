#_________________________________________________________IMPORT LIBARY________________________________________________________________#

import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import img_to_array
from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import io
import joblib
import json

#_________________________________________________________MAPPING LABELS________________________________________________________________#

label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

mixed = {

    0: 'Melanocytic nevi',
    1: 'Melanoma ',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

all_labels = [
 'Atelectasis',
 'Cardiomegaly',
 'Consolidation',
 'Edema',
 'Effusion',
 'Emphysema',
 'Fibrosis',
 'Infiltration',
 'Mass',
 'Nodule',
 'Pleural_Thickening',
 'Pneumonia',
 'Pneumothorax']

#______________________________________________________FLASK APPLICATION________________________________________________________________#

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

#__________________________________________________________MODELS ML____________________________________________________________________#

model_diabetes =joblib.load('RfModify_model_diabetes.hdf5')
model_Skin=None
model_Xray=None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # substitute in your own networks just as easily)
    global model_Skin
    tf.keras.models.load_model('Skin_Cancer.hdf5')
    global model_Xray
    model_Xray=tf.keras.models.load_model('x-ray_chest87.sav')
    
#________________________________________________________PREPARE IMAGE__________________________________________________________________#

#SKIN CANCER
def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    

    # return the processed image
    return image

# X-RAY
def prepare_image_xray(image_x, target):
    image_x = image_x.resize(target)
    image_x = image_x.convert("L")
    image_x = img_to_array(image_x)
    image_x = np.expand_dims(image_x, axis=0)
    
    # return the processed image
    return image_x
#__________________________________________________________INDEX HTML___________________________________________________________________#

@app.route("/", methods=['GET', 'POST'])
def index():
	return render_template("index.html")
@app.route("/index.html", methods=['GET', 'POST'])
def index1():
	return render_template("index.html")

#_________________________________________________________SKIN CANCER HTML______________________________________________________________#

@app.route("/skin.html", methods=['GET', 'POST'])
def Skin_Cancer():
	return render_template("skin.html")
@app.route("/skin", methods=['GET', 'POST'])
def Skin_Cancer2():
	return render_template("skin.html")

#____________________________________________________________XRAY HTML__________________________________________________________________#

@app.route("/xray.html", methods=['GET', 'POST'])
def xray():
	return render_template("xray.html")
@app.route("/xray", methods=['GET', 'POST'])
def xray1():
	return render_template("xray.html")

#__________________________________________________________DIABETES HTML________________________________________________________________#

@app.route("/diabetes.html", methods=['GET', 'POST'])
def diabetes():
	return render_template("diabetes.html")
@app.route("/diabetes", methods=['GET', 'POST'])
def diabetes1():
	return render_template("diabetes.html")

#________________________________________________________SKIN CANCER API________________________________________________________________#

@app.route('/submit', methods=["GET","POST"])
def predict():
   # initialize the data dictionary that will be returned from the
   # view
   data = {"success": False}
   # ensure an image was properly uploaded to our endpoint
   if request.method == "POST":
        if  request.files["image"]:
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(28, 28))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            results =lesion_type_dict[label_mapping[np.argmax(model_Skin.predict(image))]]
            data["predictions"] = []

            # returned prediction
            data["predictions"].append(results)

            # indicate that the request was a success
            data["success"] = True
    
        return render_template('skin.html',prediction=results)
    # return the data dictionary as a JSON response
   return flask.jsonify(data)

#________________________________________________________DIABETES API___________________________________________________________________#

@app.route('/DIABETES', methods=["GET","POST"])
def DIABETES():
    data = {"success": False}
    if request.method == "POST":
        
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
        res = model_diabetes.predict(arr)
       
        if res ==1:
            pred="Positive"
        else:
            pred = "Negative"
        
        ressdk["predict"]=pred

        data["predictions"] = []
        data["predictions"].append(pred)
        data["success"] = True
    return render_template('diabetes.html', pred=pred) , json_object

#_________________________________________________________X-RAY API_____________________________________________________________________#

@app.route('/XRAY', methods=["GET","POST"])
def XRAY():
   # initialize the data dictionary that will be returned from the
   # view
   data = {"success": False}
   
   # ensure an image was properly uploaded to our endpoint
   if request.method == "POST":
        if  request.files["image"]:
            # read the image in PIL format
            image_x = request.files["image"].read()
            image_x = Image.open(io.BytesIO(image_x))

            # preprocess the image and prepare it for classification

            image_x = prepare_image_xray(image_x, target=(128, 128))
            
            #with sess.as_default():
            #with graph.as_default():
            preds = model_Xray.predict(image_x)
        #try:
            #preds = model_Xray.predict(image_x)
            data["pre"] = []

            l={}
            for n_class, p_score in zip(all_labels, preds):
                l[n_class[:]]=p_score*100
            
            results = max(l,key=l.get)
            data["pre"].append(results)
            # returned prediction
            
            # indicate that the request was a success
            data["success"] = True
        
        #except AttributeError:
        
        return render_template('xray.html',pre=results)
    # return the data dictionary as a JSON response
   return flask.jsonify(data)

#_______________________________________________________START APPLICATION_______________________________________________________________#

if __name__ == "__main__":
    
    load_model()
    app.run(debug=True,use_reloader=False)