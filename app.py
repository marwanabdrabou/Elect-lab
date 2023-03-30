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
#from pyngrok import ngrok

#______________________________________________________Fastapi APPLICATION________________________________________________________________#

app = FastAPI()
origins = ["*"]

#______________________________________________________List of Results____________________________________________________________________#

results_skin=[]
results_diabetes=[]
results_xray=[]
results_brain=[]
#results_breast=[]


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
    preds=lesion_type_dict[label_mapping[np.argmax(model_Skin.predict(image))]]
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
        return 'Negative'
    else:
        return 'Positive'

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
    
    image = image.resize((128,128))
    image = image.convert("L")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    model_Xray= tf.keras.models.load_model('x-ray_chest87.sav')
    
    preds = model_Xray.predict(image)
    
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
    
    l={}
    for n_class, p_score in zip(all_labels, preds):
        l[n_class[:]]=p_score*100
        
    pp=max(l,key=l.get)
    results_xray.clear()
    results_xray.append(pp)
    return pp
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
    
    image = image.resize((150, 150))
    image = image.convert("L")
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
    pre=all_labels_brain[label_mapping_brain[np.argmax(model_Brain.predict(image))]]
    results_brain.clear()
    results_brain.append(pre)
    return pre

@app.get("/ResBrain")
async def get_result():
    return results_brain[0]

#______________________________________________________________Breast Cancer API___________________________________________________________________#
#@app.post('/Breast_cancer_seg')
#async def scoring_endpoint(data: UploadFile = File(...)): 
#    model_breast = SegmentationModel().model
#    model_breast.load_weights('Breast_Cancer_Unet.hdf5') 

#    image = read_imagefile(await data.read())
#    image = image.resize((128, 128))
#    image = image.convert("L")
#    image = img_to_array(image)
    
    #image = tf.io.decode_image(image) 
#    image = np.expand_dims(image, axis=0)
#    image /=255.0
#    pred = model_breast.predict(image)
#    d={"prediction": json.dumps(pred.tolist())}
#    results_breast.clear()
#    results_breast.append(d)
#    return d

#@app.get("/ResBreast")
#async def get_result():
#    return results_breast[0]

#_______________________________________________________START APPLICATION_________________________________________________________________________#

#ngrok_tunnel=ngrok.connect(8000)
#print('public URL:', ngrok_tunnel.public_url)
#nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=os.getenv("PORT", default=5000), log_level="info")
#uvicorn.run(app, port=8080)
