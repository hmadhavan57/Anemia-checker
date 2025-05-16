# Import necessary Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
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
d = st.file_uploader('Upload your Csv file')
if d is not None:
       df =  pd.read_csv(d)
       # Drpping Unwanted columns
       df.drop(columns=['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13',], inplace=True)
       
       # Formatting the Needed column
       df.rename(columns={
        '%Red Pixel' : 'Red Pixel',
        '%Green pixel' : 'Green pixel',
        '%Blue pixel' : 'Blue pixel',
       },inplace=True)
       
       # Changing the numeric to categorical columsn
       le = LabelEncoder()
       dfc = df.copy()
       for cols in df.columns:
            if dfc[cols].dtype == 'object':
                 dfc[cols] = le.fit_transform(dfc[cols])
      # Split the dataset for training
       x = dfc.drop(columns=['Number', 'Name', 'Anaemic']) 
       y = df['Anaemic']
       LR = LogisticRegression()
       LR.fit(x,y)
    
else:
        pass
# Building Navigation Bar
nav = st.sidebar.radio("Navigation",[ 'View Data','visualization', 'Prediction'])
if nav == "View Data":

    if df is not None:
        # Viewing data
        st.write('View Data')
        st.write(df)


if nav == 'visualization':

    st.write("visualization") 
    if df is not None:
         
         # Visualize all the features
         sns.histplot(data=df, x='Red Pixel', kde=True, bins=30)
         st.pyplot()
         sns.histplot(data=df, x = 'Blue pixel', kde= True, bins=30)
         st.pyplot()
         sns.histplot(data=df, x = 'Green pixel', kde=True, bins=30)
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
        red_pixel = st.number_input("Red Pixel", min_value=0.0, max_value=100.0, value=50.0)
        green_pixel = st.number_input("Green Pixel", min_value=0.0, max_value=100.0, value=50.0)
        blue_pixel = st.number_input("Blue Pixel", min_value=0.0, max_value=100.0, value=50.0)
        hb = st.number_input("Hb", min_value=0.0, max_value=20.0, value=10.0)

        input_data = pd.DataFrame({
            'Red Pixel': [red_pixel],
            'Green pixel': [green_pixel],
            'Blue pixel': [blue_pixel],
            'Hb': [hb]
        })

        prediction = LR.predict(input_data)
        if st.button('Predict'): # making button to show the result of the prediction
        
            prediction_proba = LR.predict_proba(input_data)
            if prediction[0] == 1:
                st.write("The patient is predicted to have anemia.")
            else:
                st.write("The patient is predicted to not have anemia.")
