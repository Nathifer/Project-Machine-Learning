

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar archivo CSV
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    bank = pd.read_csv(uploaded_file)

    # Mostrar el tamaño del archivo cargado
    st.write(f"Datos cargados con éxito: {bank.shape[0]} filas y {bank.shape[1]} columnas")

    # Limpiar datos (eliminar outliers, imputación, etc.)
    # Aquí agregas todo tu código de limpieza y transformación de datos

    # Imputación de valores nulos y outliers
    # Asegúrate de que las funciones de limpieza se ejecuten correctamente
    bank = remove_outliers(bank, 'age', factor=3)
    bank = remove_outliers(bank, 'balance', factor=3)
    bank = remove_outliers(bank, 'duration', factor=3)
    # Agrega aquí el resto de las transformaciones

    # Visualización de los datos
    st.subheader("Distribución de Clientes Según la Contratación de Depósitos")
    num_depositos = bank[bank['deposit'] == 'yes'].shape[0]
    num_no_depositos = bank[bank['deposit'] == 'no'].shape[0]
    
    plt.bar(["Depósitos (%d)" % num_depositos, "No Depósitos (%d)" % num_no_depositos],
            [num_depositos, num_no_depositos],
            color=["cyan", "red"],
            width=0.8)
    plt.ylabel("Número de Personas")
    plt.title("Distribución de Clientes")
    st.pyplot()

    # Mostrar una vista previa de los datos después de la limpieza
    st.write("Vista previa de los datos después de la limpieza:")
    st.dataframe(bank.head())

    # Aplicar KMeans
    st.subheader("Aplicar KMeans para Segmentación de Clientes")
    X = bank.select_dtypes(include=[np.number]).dropna()  # Seleccionar solo variables numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X_scaled)
    st.write(f"Centros de los clusters: {kmeans.cluster_centers_}")

    # Añadir la columna de predicción de clusters
    bank['Cluster'] = kmeans.labels_
    st.write("Datos con Clusters asignados:")
    st.dataframe(bank.head())


        
