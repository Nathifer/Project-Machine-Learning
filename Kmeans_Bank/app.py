import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

# Título de la aplicación
st.title("Análisis de Datos de Clientes Bancarios con KMeans")

# Cargar el modelo y el escalador
@st.cache
def load_model():
    with open('kmeans_model_bank.pkl', 'rb') as model_file:
        kmeans_model = pickle.load(model_file)
    with open('scaler_bank.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return kmeans_model, scaler

# Cargar el modelo y el escalador
kmeans_model, scaler = load_model()

# Cargar el dataset desde GitHub
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/Nathifer/Project-Machine-Learning/main/Kmeans_Bank/bank_dataset.csv"
    return pd.read_csv(url)

# Cargar los datos
bank = load_data()

# Mostrar una vista previa del DataFrame
st.subheader("Vista previa de los datos")
st.write(bank.head())

# Descripción básica
st.subheader("Estadísticas Descriptivas")
st.write(bank.describe())

# Eliminar outliers
def remove_outliers(df, column, factor=3):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Aplicar la eliminación de outliers en varias columnas
outlier_columns = ['age', 'balance', 'duration', 'pdays', 'previous']
for col in outlier_columns:
    bank = remove_outliers(bank, col)

# Imputar valores nulos (ya lo has hecho en tu código)
# Asegúrate de que no haya valores nulos
bank = bank.fillna(method='ffill')  # O puedes hacer la imputación que hayas utilizado en el código

# Mostrar datos sin valores nulos
st.subheader("Datos sin valores nulos")
st.write(bank.isnull().sum())

# Oversampling para balancear las clases
X = bank.drop(columns=['deposit'])
y = bank['deposit']

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Verificar el balance de las clases después del oversampling
num_depositos_resampled = y_resampled.value_counts()['yes']
num_no_depositos_resampled = y_resampled.value_counts()['no']

st.subheader("Balance de clases después de oversampling")
st.write(f"Depósitos: {num_depositos_resampled}")
st.write(f"No Depósitos: {num_no_depositos_resampled}")

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

st.subheader("Tamaño de los conjuntos de entrenamiento y prueba")
st.write(f"Tamaño del conjunto de entrenamiento (X): {X_train.shape}")
st.write(f"Tamaño del conjunto de prueba (X): {X_test.shape}")
st.write(f"Tamaño del conjunto de entrenamiento (y): {y_train.shape}")
st.write(f"Tamaño del conjunto de prueba (y): {y_test.shape}")

# Codificación de variables categóricas (Label Encoding)
binary_columns = ['default', 'housing', 'loan']

label_encoder = LabelEncoder()
for column in binary_columns:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.transform(X_test[column])

st.subheader("Ejemplo de datos después de Label Encoding")
st.write(X_train.head())

# Codificación OneHotEncoding
encoder = OneHotEncoder(sparse_output=False)
columns_to_encode = ["poutcome", "marital", "education", "contact", "month", "job"]

one_hot_encoded_array_train = encoder.fit_transform(X_train[columns_to_encode])
one_hot_encoded_array_test = encoder.transform(X_test[columns_to_encode])

one_hot_encoded_df_train = pd.DataFrame(one_hot_encoded_array_train, columns=encoder.get_feature_names_out(columns_to_encode))
one_hot_encoded_df_test = pd.DataFrame(one_hot_encoded_array_test, columns=encoder.get_feature_names_out(columns_to_encode))

X_train = pd.concat([X_train.reset_index(drop=True), one_hot_encoded_df_train.reset_index(drop=True)], axis=1).drop(columns_to_encode, axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), one_hot_encoded_df_test.reset_index(drop=True)], axis=1).drop(columns_to_encode, axis=1)

st.subheader("Datos después de OneHotEncoding")
st.write(X_train.head())

# Realizar la predicción utilizando el modelo KMeans
st.subheader("Predicción de Clústeres con KMeans")
input_data = X_test.iloc[0:5]  # Ejemplo con las primeras 5 filas del conjunto de prueba

# Escalar los datos de entrada
input_data_scaled = scaler.transform(input_data)

# Predecir los clústeres
cluster_predictions = kmeans_model.predict(input_data_scaled)

st.write("Clústeres predichos para las primeras 5 filas del conjunto de prueba:")
st.write(cluster_predictions)

# Gráfica de predicciones
st.subheader("Visualización de Clústeres")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_test['age'], X_test['balance'], c=cluster_predictions, cmap='viridis', s=50, alpha=0.5)
ax.set_xlabel('Edad')
ax.set_ylabel('Balance')
ax.set_title('Clústeres Predichos')
st.pyplot(fig)
