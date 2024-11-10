import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Cargar el modelo KMeans y el escalador desde archivos pickle
@st.cache_resource
def load_model():
    with open('kmeans_model_bank.pkl', 'rb') as model_file:
        kmeans_model = pickle.load(model_file)
    with open('scaler_bank.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return kmeans_model, scaler

# Función para cargar el dataset desde GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Nathifer/Project-Machine-Learning/main/Kmeans_Bank/bank_dataset.csv"
    data = pd.read_csv(url)
    return data
    
# Función para predecir el clúster basándonos en las entradas
def predict_cluster(kmeans_model, scaler, input_values, bank_encoded, numeric_columns):
    # Crear un DataFrame con las mismas columnas que el modelo espera
    input_data = dict(zip(bank_encoded.columns, input_values))
    input_df = pd.DataFrame([input_data])

    # Asegúrate de que las columnas numéricas sean del tipo adecuado
    for col in numeric_columns:
        input_df[col] = input_df[col].astype(float)

    # Asegúrate de que las columnas categóricas estén codificadas
    categorical_columns = ['job', 'marital', 'education', 'month', 'contact', 'poutcome']
    for col in categorical_columns:
        input_df[col] = input_df[col].astype('category')

    # Escalar las entradas
    scaled_input = scaler.transform(input_df)

    # Hacer la predicción con el modelo KMeans
    predicted_cluster = kmeans_model.predict(scaled_input)
    
    return predicted_cluster

# Función para mostrar el gráfico de los clústeres
def show_cluster_plot(kmeans_model, scaler, data):
    # Predecir los clústeres
    data_scaled = scaler.transform(data)
    clusters = kmeans_model.predict(data_scaled)
    data['Cluster'] = clusters

    # Graficar los resultados (2D)
    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='viridis')
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_title("Visualización de Clústeres")

    # Añadir leyenda
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

# Función para mostrar los primeros datos
def show_data(bank):
    st.write("### Datos del Banco")
    st.write(bank.head())

# Función para mostrar estadísticas descriptivas
def show_statistics(bank):
    st.write("### Estadísticas Descriptivas")
    st.write(bank.describe())

# Función principal para la interfaz de Streamlit
def main():
    st.title('Predicción de Clústeres - Análisis de Datos del Banco')

    # Cargar los modelos y el escalador
    kmeans_model, scaler = load_model()

    # Cargar los datos
    bank = load_data()

    # Crear formulario para ingresar parámetros
    st.sidebar.header("Ingresa los parámetros de las variables")
    job = st.sidebar.selectbox("Trabajo", bank['job'].unique())
    marital = st.sidebar.selectbox("Estado Civil", bank['marital'].unique())
    education = st.sidebar.selectbox("Educación", bank['education'].unique())
    month = st.sidebar.selectbox("Mes de Contacto", bank['month'].unique())  
    contact = st.sidebar.selectbox("Tipo de Contacto", bank['contact'].unique())
    poutcome = st.sidebar.selectbox("Resultado de Campaña Anterior", bank['poutcome'].unique())

    # Codificar las variables categóricas
    input_data = {
        'job': job,
        'marital': marital,
        'education': education,
        'month': month,
        'contact': contact,
        'poutcome': poutcome
    }

    # Codificar las entradas utilizando el encoder
    bank_encoded = bank.copy()
    categorical_columns = ['job', 'marital', 'education', 'month', 'contact', 'poutcome']

    # Asegurarse de que las columnas categóricas sean del tipo 'category'
    for column in categorical_columns:
        bank_encoded[column] = bank_encoded[column].astype('category')

    # Codificar las entradas usando la posición de las categorías
    input_values = []
    for col in categorical_columns:
        input_values.append(bank_encoded[col].cat.categories.get_loc(input_data[col]))

    # Obtener las variables numéricas
    numeric_columns = [col for col in bank.columns if col not in categorical_columns and bank[col].dtype != 'object']
    numeric_values = [float(st.sidebar.text_input(f"Ingrese valor para {col}", 0)) for col in numeric_columns]

    # Concatenar las características numéricas y categóricas
    all_input_values = input_values + numeric_values

    # Predecir el clúster
    predicted_cluster = predict_cluster(kmeans_model, scaler, all_input_values, bank_encoded, numeric_columns)  
    st.sidebar.write(f"**Clúster Predicho:** {predicted_cluster[0]}")

    # Mostrar el gráfico
    show_cluster_plot(kmeans_model, scaler, bank_encoded)

    # Crear una barra lateral para navegación
    option = st.sidebar.selectbox(
        'Selecciona una opción:',
        ['Ver Datos', 'Estadísticas Descriptivas']
    )

    # Mostrar los datos seleccionados
    if option == 'Ver Datos':
        show_data(bank)
    elif option == 'Estadísticas Descriptivas':
        show_statistics(bank)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
