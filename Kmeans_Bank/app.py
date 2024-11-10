

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Clustering con K-means")

# Cargar los datos
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Mostrar las primeras filas del dataset
    st.write("Primeras filas del dataset:", df.head())

    # Preprocesamiento y selección de características
    X = df.select_dtypes(include=[float, int])  # Usamos solo las columnas numéricas

    # Escalado de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determinación del número óptimo de clusters (Método del Codo y Silueta)
    K_range = range(2, 11)
    inertia_values = []
    silhouette_values = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        
        inertia_values.append(kmeans.inertia_)
        
        # Coeficiente de silueta
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_values.append(silhouette_avg)
    
    # Mostrar los gráficos del codo y la silueta
    st.subheader("Método del Codo")
    fig, ax = plt.subplots()
    ax.plot(K_range, inertia_values, marker='o')
    ax.set_title("Método del Codo")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Inercia")
    st.pyplot(fig)

    st.subheader("Coeficiente de Silueta")
    fig, ax = plt.subplots()
    ax.plot(K_range, silhouette_values, marker='o', color='orange')
    ax.set_title("Coeficiente de Silueta")
    ax.set_xlabel("Número de Clusters (K)")
    ax.set_ylabel("Coeficiente de Silueta")
    st.pyplot(fig)

    # Selección de K óptimo
    k_optimo = st.slider("Selecciona el número de clusters (K)", 2, 10, 3)

    # Aplicar KMeans con el valor seleccionado de K
    kmeans = KMeans(n_clusters=k_optimo, random_state=0)
    kmeans.fit(X_scaled)
    
    # Asignar los clusters
    df['Cluster'] = kmeans.labels_

    # Reducir la dimensionalidad con PCA para visualización
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)
    df_reduced = pd.DataFrame(X_reduced, columns=['Componente 1', 'Componente 2'])
    df_reduced['Cluster'] = df['Cluster']

    # Visualización de los clusters
    st.subheader("Visualización de Clusters")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_reduced, x='Componente 1', y='Componente 2', hue='Cluster', palette='viridis', s=100, alpha=0.7, edgecolor='w')
    ax.set_title("Visualización de Clusters")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    st.pyplot(fig)

else:
    st.warning("Por favor, sube un archivo CSV para continuar.")

        