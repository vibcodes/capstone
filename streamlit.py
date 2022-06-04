#please run this code on streamlit
import streamlit as st
  
st.title('Capstone Project-Pneumonia Detection')


image = st.file_uploader('Upload your xray scan here')

import zipfile
with zipfile.ZipFile('best_model.zip', 'r') as zip_ref:
    zip_ref.extractall()
import os
st.text(os.listdir('best_model/'))
from tensorflow.keras.models import load_model
pred_model=load_model('best_model')
import requests, io
url = image
response = requests.get(url)
img = Image.open(io.BytesIO(response.content))
st.image(img)
img = img.resize((224, 224))  # Resize image
img = img.convert('RGB')

# Convert to float32 array
img = preprocess_input(np.array(img, dtype=np.float32))
img=np.reshape(img,(1,224,224,3))
pred=pred_model.predict(img)
st.image(pred[0]>0.2)
pred[0]=np.where(pred[0]>0.2,1,0)
if pred[0].sum()>0:
  st.text('Pneumonia Detected')
else:
  st.text('No Pneumonia Detected')
