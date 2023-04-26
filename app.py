#_________________________________________________________IMPORT LIBARY________________________________________________________________#

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO
from model_definition import SegmentationModel
import joblib
from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import os
from pyngrok import ngrok

#______________________________________________________Fastapi APPLICATION________________________________________________________________#

app = FastAPI()
origins = ["*"]

#______________________________________________________List of Results____________________________________________________________________#

results_skin=[]
results_diabetes=[]
results_xray=[]
results_brain=[]
results_breast=[]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#______________________________________________________Function to read File________________________________________________________________#

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

#____________________________________________________________Skin API________________________________________________________________________#

@app.post("/skin_cancer_predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    
    image = read_imagefile(await file.read())
    # image = Image.open(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize((28,28))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    
    model_Skin= tf.keras.models.load_model("skin_cancer_model.sav")
    
    label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'}

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'}

    mixed = {
        0: 'Melanocytic nevi',
        1: 'Melanoma ',
        2: 'Benign keratosis-like lesions',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratoses',
        5: 'Vascular lesions',
        6: 'Dermatofibroma'}
    preds={"result": lesion_type_dict[label_mapping[np.argmax(model_Skin.predict(image))]]}
    results_skin.clear()
    results_skin.append(preds)
    return preds

@app.get("/ResSkin")
async def get_result():
    return results_skin[0]

#___________________________________________________________Diabetes API_______________________________________________________________________#

class model_input_Diabetes(BaseModel):
    
    Polydipsia:int
    Polyuria:int
    Gender:int
    Age:int
    Sudden_Weight_Loss:int
    Alopecea:int

model_diabetes =joblib.load('RfModify_model_diabetes.hdf5')

def diabetes_predd(input_parameters : model_input_Diabetes):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    Polydipsia = input_dictionary['Polydipsia']
    Polyuria = input_dictionary['Polyuria']
    Gender = input_dictionary['Gender']
    Age = input_dictionary['Age']
    Sudden_Weight_Loss = input_dictionary['Sudden_Weight_Loss']
    Alopecea = input_dictionary['Alopecea']
    
    
    input_list = [ Polydipsia, Polyuria,Gender,Age,  Sudden_Weight_Loss, Alopecea]
    
    prediction = model_diabetes.predict([input_list])
    
    if (prediction[0] == 0):
        return {"result":'Negative'}
    else:
        return {"result":'Positive'}

@app.post("/diabetes_prediction")
async def get_predict(input : model_input_Diabetes):
    data=diabetes_predd(input)
    results_diabetes.clear()
    results_diabetes.append(data)
    return data

@app.get("/ResDia")
async def get_result():
    return results_diabetes[0]

#______________________________________________________________X-Ray API_______________________________________________________________________#

@app.post('/chest_predict')
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    
    image = read_imagefile(await file.read())
    
    image = image.resize((150, 150))
    image = image.convert("L")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    model_chest =tf.keras.models.load_model('pneumonia.sav')
    
    #chest Tumor labels
    label_mapping_chest = {
        0: "Pneumonia",
        1:"Normal"}

    all_labels_chest = {
     "Pneumonia":'Pneumonia',
     "Normal" :'Normal'}
    
    pre={"result":all_labels_chest[label_mapping_chest[np.argmax(model_chest.predict(image))]]}
    results_xray.clear()
    results_xray.append(pre)
    return pre


@app.get("/ResXray")
async def get_result():
    return results_xray[0]

#______________________________________________________________Brain API_______________________________________________________________________#

@app.post("/brain_tumor_predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    
    image = read_imagefile(await file.read())
    
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    model_Brain =tf.keras.models.load_model('Brain_Tumor.hdf5')
    
    #Brain Tumor labels
    label_mapping_brain = {
        0: "glioma_tumor",
        2:"meningioma_tumor",
        1:"no_tumor",
        3:"pituitary_tumor"}

    all_labels_brain = {
     "glioma_tumor":'Giloma Tumor',
     "no_tumor" :'No Tumor',
     "meningioma_tumor" :'Meningioma Tumor',
     "pituitary_tumor" :'Pituitary Tumor'}
    pre={"result": all_labels_brain[label_mapping_brain[np.argmax(model_Brain.predict(image))]]}
    results_brain.clear()
    results_brain.append(pre)
    return pre

@app.get("/ResBrain")
async def get_result():
    return results_brain[0]

#______________________________________________________________Breast Cancer API___________________________________________________________________#
@app.post('/Breast_cancer')
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    
    image = read_imagefile(await file.read())
    
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    model_breast =tf.keras.models.load_model('best_model_2.hdf5')
    
    
    label_mapping_breast = {
        0: "Benign",
        1:"malignant",
        2:"normal"}

    all_labels_breast = {
     "Benign":'Benign',
     "malignant" :'malignant',
     "normal" :'normal'}

    pre={"result":all_labels_breast[label_mapping_breast[np.argmax(model_breast.predict(image))]]}
    results_breast.clear()
    results_breast.append(pre)
    return pre

@app.get("/ResBreast")
async def get_result():
    return results_breast[0]

#_______________________________________________________START APPLICATION_________________________________________________________________________#

#ngrok_tunnel=ngrok.connect(8000)
#print('public URL:', ngrok_tunnel.public_url)
#nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", default=5000), log_level="info")
#uvicorn.run(app, port=8080)
