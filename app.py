

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Rutas de los modelos y el escalador
model_path = 'kmeans_model_bank.pkl'  # Ruta del modelo
scaler_path = 'scaler_bank.pkl'        # Ruta del escalador

# Cargar el modelo y el escalador
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    st.error(f"Error: {e}")
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")

# Interfaz de usuario
st.title("Predicción de Clientes del Banco")

# Entradas del usuario
job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
education = st.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox('Credit Default?', ['yes', 'no'])
housing = st.selectbox('Housing Loan?', ['yes', 'no'])
loan = st.selectbox('Personal Loan?', ['yes', 'no'])
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
poutcome = st.selectbox('Previous Outcome', ['failure', 'nonexistent', 'success'])
age = st.number_input('Age', min_value=18, max_value=100, step=1)
balance = st.number_input('Balance', min_value=0, step=100)
duration = st.number_input('Duration', min_value=0, step=10)
pdays = st.number_input('Pdays', min_value=-1, step=1)
campaign = st.number_input('Campaign', min_value=1, step=1)

# Preparar los datos para la predicción
# Crear DataFrame con las columnas categóricas y numéricas
input_data = pd.DataFrame({
    'job': [job], 'marital': [marital], 'education': [education],
    'default': [default], 'housing': [housing], 'loan': [loan],
    'contact': [contact], 'poutcome': [poutcome],
    'age': [age], 'balance': [balance], 'duration': [duration],
    'pdays': [pdays], 'campaign': [campaign]
})

# Separar columnas categóricas y numéricas
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
numerical_columns = ['age', 'balance', 'duration', 'pdays', 'campaign']

# Aplicar One-Hot Encoding a las columnas categóricas
try:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_categorical_data = encoder.fit_transform(input_data[categorical_columns])
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

    # Combinar con las columnas numéricas
    input_data_final = pd.concat([encoded_categorical_df, input_data[numerical_columns]], axis=1)

    # Escalar las columnas numéricas
    input_data_final[numerical_columns] = scaler.transform(input_data_final[numerical_columns])
except Exception as e:
    st.error(f"Error al procesar los datos: {e}")

# Realizar predicción
if st.button('Realizar Predicción'):
    try:
        prediction = model.predict(input_data_final)
        if prediction[0] == 1:
            st.success('El cliente probablemente suscriba el préstamo.')
        else:
            st.warning('El cliente probablemente no suscriba el préstamo.')
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        
