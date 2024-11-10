import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
@st.cache_data  # Esto asegura que el archivo se cargue solo una vez
def load_data():
    data = pd.read_csv('bank_dataset.csv')  # Asegúrate de que el archivo esté en la ruta correcta
    return data

# Función para mostrar el dataset
def show_data():
    st.write("### Datos del Banco")
    st.write(bank.head())  # Muestra las primeras filas del dataset

# Función para mostrar estadísticas descriptivas
def show_statistics():
    st.write("### Estadísticas Descriptivas")
    st.write(bank.describe())

# Función para mostrar un gráfico de dispersión
def show_scatter_plot():
    st.write("### Gráfico de Edad vs. Balance")
    fig, ax = plt.subplots()
    sns.scatterplot(data=bank, x='age', y='balance', ax=ax)
    st.pyplot(fig)

# Función principal para la interfaz de Streamlit
def main():
    st.title('Análisis de Datos del Banco')

    # Cargar datos
    bank = load_data()

    # Crear una barra lateral para navegación
    option = st.sidebar.selectbox(
        'Selecciona una opción:',
        ['Ver Datos', 'Estadísticas Descriptivas', 'Gráfico de Edad vs. Balance']
    )

    # Mostrar los datos seleccionados
    if option == 'Ver Datos':
        show_data()
    elif option == 'Estadísticas Descriptivas':
        show_statistics()
    elif option == 'Gráfico de Edad vs. Balance':
        show_scatter_plot()

if __name__ == '__main__':
    main()
