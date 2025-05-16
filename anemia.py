# Import necessary Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set the custom theme for viz
sns.set_style("dark")
sns.set(rc={"axes.facecolor":"#7faacb","figure.facecolor":"#7f92cb"})
sns.set_context("poster",font_scale = .7)

# Upload the dataset
st.title('Anemia Checker')
st.write('View Data')
df = None
d = st.file_uploader('Upload your CSV file')
if d is not None:
    df = pd.read_csv(d)
    # Dropping Unwanted columns
    df.drop(columns=['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13'], inplace=True)
    
    # Formatting the Needed column
    df.rename(columns={
        '%Red Pixel': 'Red Pixel',
        '%Green pixel': 'Green pixel',
        '%Blue pixel': 'Blue pixel',
    }, inplace=True)
    
    # Changing the numeric to categorical columns
    le = LabelEncoder()
    dfc = df.copy()
    for cols in df.columns:
        if dfc[cols].dtype == 'object':
            dfc[cols] = le.fit_transform(dfc[cols])
    
    # Split the dataset for training
    x = dfc.drop(columns=['Number', 'Name', 'Anaemic']) 
    y = dfc['Anaemic']  # Use the encoded 'Anaemic' column
    LR = LogisticRegression()
    LR.fit(x, y)
    
else:
    st.write("Please upload a CSV file to proceed.")

# Building Navigation Bar
nav = st.sidebar.radio("Navigation", ['View Data', 'Visualization', 'Prediction'])
if nav == "View Data":
    if df is not None:
        # Viewing data
        st.write('View Data')
        st.write(df)

if nav == 'Visualization':
    st.write("Visualization") 
    if df is not None:
        # Visualize all the features
        sns.histplot(data=df, x='Red Pixel', kde=True, bins=30)
        st.pyplot()
        sns.histplot(data=df, x='Blue pixel', kde=True, bins=30)
        st.pyplot()
        sns.histplot(data=df, x='Green pixel', kde=True, bins=30)
        st.pyplot()
        sns.kdeplot(data=df, x='Hb')
        st.pyplot()
        sns.countplot(data=df, x='Anaemic')
        st.pyplot()

if nav == "Prediction":
    st.write('Prediction')
    if df is not None:
        # Get the data from the user for new prediction
        st.write("Please input the following values to predict anemia:")
        upload_image = st.file_uploader('Upload an eye image', type=['jpg', 'png', 'jpeg'])
        hb = st.number_input("Hb", min_value=0.0, max_value=20.0, value=10.0)

        if upload_image is not None:
            file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            R, G, B = cv2.split(img)
            total_pixels = img.shape[0] * img.shape[1]
            total_intensity = total_pixels * 255
            R_percentage = (np.sum(R) / total_intensity) * 100
            G_percentage = (np.sum(G) / total_intensity) * 100
            B_percentage = (np.sum(B) / total_intensity) * 100
            
            input_data = pd.DataFrame({
                'Red Pixel': [R_percentage],
                'Green pixel': [G_percentage],
                'Blue pixel': [B_percentage],
                'Hb': [hb]
            })
            
            if st.button('Predict'):
                prediction = LR.predict(input_data)
                prediction_proba = LR.predict_proba(input_data)
                if prediction[0] == 1:
                    st.write("The patient is predicted to have anemia.")
                else:
                    st.write("The patient is predicted to not have anemia.")
                st.write(f"Prediction Probability: {prediction_proba}")
        else:
            st.write("Please upload an image to proceed.")
    else:
        st.write("Please upload a CSV file to proceed.")