# Application Streamlit de prédiction du prix d'un trajet de taxi
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Taxi Trip Price Prediction", page_icon="🚕", layout="centered")


# Charger les objets sauvegardés (une seule fois grâce au cache)
@st.cache_resource
def load_objects():
    # importer les encodeurs
    encoders = joblib.load('encoders.joblib')
    # importer le best modèle
    xgb_model = joblib.load('xgb_model.joblib')
    # importer les valeurs uniques
    uniques = joblib.load('uniques.joblib')
   #normalisateur 
    scaler = joblib.load('scaler.joblib')  
    return encoders, xgb_model, uniques,scaler


encoders, xgb_model, uniques, scaler = load_objects()

 
# Fonction de prédiction
def Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
              Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes):
    # Encoder les valeurs des Time_of_Day, Day_of_Week, Traffic_Conditions et Weather
    Time_of_Day = encoders[0].transform([Time_of_Day])[0]
    Day_of_Week = encoders[1].transform([Day_of_Week])[0]
    Traffic_Conditions = encoders[2].transform([Traffic_Conditions])[0]
    Weather = encoders[3].transform([Weather])[0]
    # vecteur des valeurs numériques
    x_new = np.array([Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                      Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes], dtype=float)
    x_new = x_new.reshape(1, -1)  # convertir en un 2D array


    # #NORMALISER 
    x_new = scaler.transform(x_new)
    #Prédire
    y_pred = xgb_model.predict(x_new)
    # Arrondir
    return round(float(y_pred[0]), 2)


# Fonction de prédiction multiple
def Pred_func_csv(df):
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :10].values:
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
        predictions.append(y_pred)
    df['Trip_Price'] = predictions
    return df


# Titre de l'application
st.title("🚕 Taxi Trip Price Prediction")

# Créer les onglets
tab1, tab2 = st.tabs(['Simple Prédiction', 'Prédiction multiple'])

# ---------------- Onglet 1 : prédiction simple ----------------
with tab1:
    st.subheader("Predict the price of a taxi trip with a single input")
    st.write("""This machine learning model allows us to predict the price of a taxi trip
             from the trip distance, time of day, day of week, passenger count, traffic conditions, weather,
             base fare, per km rate, per minute rate and trip duration.""")

    with st.form('form_simple'):
        col1, col2 = st.columns(2)
        with col1:
            Trip_Distance_km = st.number_input('Trip Distance (km)', min_value=0.0, value=25.0, step=0.5)
            Time_of_Day = st.selectbox('Time of Day', options=uniques[0])
            Day_of_Week = st.selectbox('Day of Week', options=uniques[1])
            Passenger_Count = st.number_input('Passenger Count', min_value=1, max_value=4, value=2, step=1)
            Traffic_Conditions = st.selectbox('Traffic Conditions', options=uniques[2])
        with col2:
            Weather = st.selectbox('Weather', options=uniques[3])
            Base_Fare = st.number_input('Base Fare', min_value=0.0, value=3.5, step=0.1)
            Per_Km_Rate = st.number_input('Per Km Rate', min_value=0.0, value=1.2, step=0.05)
            Per_Minute_Rate = st.number_input('Per Minute Rate', min_value=0.0, value=0.3, step=0.01)
            Trip_Duration_Minutes = st.number_input('Trip Duration (minutes)', min_value=0.0, value=60.0, step=1.0)
        submit = st.form_submit_button('Predict')

    if submit:
        y_pred = Pred_func(Trip_Distance_km, Time_of_Day, Day_of_Week, Passenger_Count, Traffic_Conditions,
                           Weather, Base_Fare, Per_Km_Rate, Per_Minute_Rate, Trip_Duration_Minutes)
        st.success(f"Trip Price : {y_pred} $")

# ---------------- Onglet 2 : prédiction multiple ----------------
with tab2:
    st.subheader("Predict the price of a taxi trip with multiple inputs")
    st.write("""This machine learning model allows us to predict the price of a taxi trip
             from the trip distance, time of day, day of week, passenger count, traffic conditions, weather,
             base fare, per km rate, per minute rate and trip duration.""")

    file = st.file_uploader('Upload a csv file', type=['csv'])

    if file is not None:
        # Lire le fichier csv
        df = pd.read_csv(file)
        st.write("Aperçu des données chargées :")
        st.dataframe(df.head())

        if st.button('Predict'):
            df_pred = Pred_func_csv(df)
            st.write("Résultats des prédictions :")
            st.dataframe(df_pred)
            # Télécharger le fichier des prédictions
            st.download_button(label='Download a csv file',
                               data=df_pred.to_csv(index=False).encode('utf-8'),
                               file_name='predictions.csv',
                               mime='text/csv')
