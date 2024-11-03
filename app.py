import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Rutas de los modelos y el escalador
model_path = 'kmeans_model_bank.pkl'  # Ruta del modelo
scaler_path = 'scaler_bank.pkl'        # Ruta del escalador
encoder_path = 'encoder_bank.pkl'      # Ruta del codificador

# Cargar el modelo, el escalador y el codificador
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)  # Cargar el codificador
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()  # Detener la ejecución si hay un error al cargar modelos
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}")
    st.stop()

# Crear un LabelEncoder para las columnas binarias
label_encoder = LabelEncoder()

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
month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
age = st.number_input('Age', min_value=18, max_value=100, step=1)
balance = st.number_input('Balance', min_value=0, step=100)
duration = st.number_input('Duration', min_value=0, step=10)
pdays = st.number_input('Pdays', min_value=-1, step=1)
campaign = st.number_input('Campaign', min_value=1, step=1)

# Preparar los datos para la predicción
input_data = pd.DataFrame({
    'job': [job], 
    'marital': [marital], 
    'education': [education],
    'default': [default], 
    'housing': [housing], 
    'loan': [loan],
    'contact': [contact], 
    'poutcome': [poutcome],
    'month': [month],  
    'age': [age], 
    'balance': [balance], 
    'duration': [duration],
    'pdays': [pdays], 
    'campaign': [campaign]
})

# Aplicar Label Encoding a las columnas binarias
binary_columns = ['default', 'housing', 'loan']

for column in binary_columns:
    # Asegúrate de que las columnas se codifiquen correctamente
    try:
        input_data[column] = label_encoder.fit_transform(input_data[column])  # Ajustar el encoder y transformar
    except ValueError as e:
        st.error(f"Error al codificar la columna {column}: {e}")

# Aplicar One-Hot Encoding a las columnas categóricas
categorical_columns = ['job', 'marital', 'education', 'contact', 'poutcome', 'month']
encoded_categorical_data = encoder.transform(input_data[categorical_columns])
encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenar con las columnas numéricas
numerical_columns = ['age', 'balance', 'duration', 'pdays', 'campaign']
input_data_final = pd.concat([encoded_categorical_df.reset_index(drop=True), input_data[numerical_columns]], axis=1)

# Escalar las columnas numéricas
input_data_final[numerical_columns] = scaler.transform(input_data_final[numerical_columns])

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
