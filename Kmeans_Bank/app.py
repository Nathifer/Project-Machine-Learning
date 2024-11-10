import pandas as pd
import streamlit as st

# Función para cargar el dataset desde GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Nathifer/Project-Machine-Learning/main/Kmeans_Bank/bank_dataset.csv"
    data = pd.read_csv(url)
    return data

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
    st.title('Análisis de Datos del Banco')

    # Cargar los datos desde GitHub
    bank = load_data()

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
