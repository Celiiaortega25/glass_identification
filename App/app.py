import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
import sys
sys.tracebacklimit = 0

st.image('../../../../../Downloads/download.jpg')          
st.title("Model to predict glass types")


    # loading the model
models_path = '../Models/'
model_name = models_path + 'best_rf.pkl'
loaded_model = pickle.load(open(model_name, 'rb'))

    # loading the transformers
transformers_path = '../Transformers/'
transformer_name = transformers_path + 'transformer.pkl'
loaded_transformer = pickle.load(open(transformer_name, 'rb'))

    # loading the scaler
scalers_path = '../Scalers/'
scalers_name = scalers_path + 'scaler.pkl'
loaded_scaler = pickle.load(open(scalers_name, 'rb'))
           
    # Lists of accptable values
    
RI_slider = st.slider("RI:", min_value=1.51115, max_value=1.53393, step=0.01)
Na_slider = st.slider("Na:", min_value=10.73, max_value=17.38, step=0.01)
Mg_slider = st.slider("Mg:", min_value=0.0, max_value=4.49, step=0.01)
Al_slider = st.slider("Al:", min_value=0.29, max_value=3.5, step=0.01)
Si_slider = st.slider("Si:", min_value=69.81, max_value=75.41, step=0.01)
K_slider = st.slider("K:", min_value=0.0, max_value=6.21, step=0.01)
Ca_slider = st.slider("Ca:", min_value=5.43, max_value=16.19, step=0.01)
Ba_slider = st.slider("Ba:", min_value=0.0, max_value=3.15, step=0.01)
Fe_slider = st.slider("Fe:", min_value=0.0, max_value=0.51, step=0.01)

               
    # when 'Predict' is clicked, make the prediction and store it 
if st.button("Get Your Prediction"): 

    input_data = {
        'RI': [RI_slider],
        'Na': [Na_slider],
        'Mg': [Mg_slider],
        'Al': [Al_slider],
        'Si': [Si_slider],
        'K': [K_slider],
        'Ca': [Ca_slider],
        'Ba': [Ba_slider],
        'Fe': [Fe_slider]
        }

    input_df = pd.DataFrame(input_data)

        # Preprocess the input data

    X_transformed = loaded_transformer.transform(input_df)
    X_transformed_df = pd.DataFrame(X_transformed, columns = input_df.columns)

    X_scaled = loaded_scaler.transform(X_transformed_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns = input_df.columns)

    prediction = loaded_model.predict(X_scaled_df)
    prediction_probs = loaded_model.predict_proba(X_scaled_df)  
    
    predicted_class = prediction.item(0)
    predicted_probability = prediction_probs.item(0, np.argmax(prediction_probs))

    glass_types = {
        1: 'Building windows float processed',
        2: 'Building windows non float processed',
        3: 'Vehicle windows float processed',
        5: 'Containers',
        6: 'Tableware',
        7: 'Headlamps'
    }

    if predicted_class in glass_types:
        glass_type = int(predicted_class)
        glass_name = glass_types[glass_type]
        st.success("The model predicts glass type {} - '{}'".format(glass_type, glass_name, predicted_probability))
    else:
        st.error("Invalid glass type prediction")


        
