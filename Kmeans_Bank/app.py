import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Cargar el modelo KMeans y el escalador desde archivos pickle
@st.cache_data
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
def predict_cluster(kmeans_model, scaler, inputs):
    # Verifica la forma de inputs para asegurarte de que tiene el mismo número de características que los datos con los que se entrenó el scaler
    expected_features = scaler.transform([inputs]).shape[1]
    
    if len(inputs) != expected_features:
        raise ValueError(f"El número de características en 'inputs' debe ser {expected_features}. Se recibió {len(inputs)} características.")
    
    # Si las características coinciden, transformamos los datos de entrada
    scaled_input = scaler.transform(np.array([inputs]))
    
    # Hacemos la predicción con el modelo KMeans
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
    age = st.sidebar.number_input("Edad", min_value=18, max_value=100, value=30)
    job = st.sidebar.selectbox("Trabajo", bank['job'].unique())
    marital = st.sidebar.selectbox("Estado Civil", bank['marital'].unique())
    education = st.sidebar.selectbox("Educación", bank['education'].unique())
    balance = st.sidebar.number_input("Balance")
    housing = st.sidebar.selectbox("Vivienda", bank['housing'].unique())
    loan = st.sidebar.selectbox("Préstamo", bank['loan'].unique())
    contact = st.sidebar.selectbox("Tipo de Contacto", bank['contact'].unique())
    poutcome = st.sidebar.selectbox("Resultado de Campaña Anterior", bank['poutcome'].unique())

    # Codificar las variables categóricas
    input_data = {
        'job': job,
        'marital': marital,
        'education': education,
        'contact': contact,
        'poutcome': poutcome,
        'month' : month
    } 

    # Aquí es necesario transformar las variables categóricas antes de hacer la predicción
    bank_encoded = bank.copy()

    # Asegurarse de que las columnas categóricas sean del tipo 'category'
    for column in ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']:
        bank_encoded[column] = bank_encoded[column].astype('category')

        # Verificar si el valor ingresado está en las categorías disponibles
        if input_data[column] not in bank_encoded[column].cat.categories:
            st.sidebar.error(f"El valor '{input_data[column]}' para {column} no es válido.")
            return

    # Transformar las entradas
    input_values = np.array([list(input_data.values())])
    for i, col in enumerate(input_data.keys()):
        if col in ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']:
            # Asegurarse de que las columnas sean del tipo 'category' y luego acceder a las categorías
            input_values[0, i] = bank_encoded[col].cat.categories.get_loc(input_data[col])

    # Predecir el clúster
    predicted_cluster = predict_cluster(kmeans_model, scaler, input_values[0])
    st.sidebar.write(f"**Clúster Predicho:** {predicted_cluster}")

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

if __name__ == '__main__':
    main()
