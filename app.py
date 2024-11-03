
import streamlit as st
import joblib
import numpy as np

# Rutas de los modelos y el escalador
model_path = '/content/drive/My Drive/kmeans_model_bank.pkl'  # Ruta del modelo
scaler_path = '/content/drive/My Drive/scaler_bank.pkl'        # Ruta del escalador

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

st.write("Introduce los datos para hacer una predicción:")
# Entradas del usuario
job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
education = st.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox('Credit Default?', ['yes', 'no'])
housing = st.selectbox('Housing Loan?', ['yes', 'no'])
loan = st.selectbox('Personal Loan?', ['yes', 'no'])
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
poutcome = st.selectbox('Previous Outcome', ['failure', 'nonexistent', 'success'])

# Botón para realizar la predicción
if st.button('Realizar Predicción'):
    # Prepara los datos de entrada para la predicción
    input_data = np.array([[job, marital, education, default, housing, loan, contact, poutcome]])

    # Escalar los datos
    input_data_scaled = scaler.transform(input_data)

    # Realiza la predicción
    prediction = model.predict(input_data_scaled)

    # Muestra el resultado
    if prediction[0] == 1:
        st.success('El cliente probablemente suscriba el préstamo.')
    else:
        st.warning('El cliente probablemente no suscriba el préstamo.')
    