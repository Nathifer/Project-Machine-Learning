import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Definir la función para eliminar outliers usando el IQR
def remove_outliers(df, column, factor=3):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Función para imputar valores nulos en 'age', 'education' y 'marital' con la media y moda
def imputar_valores_nulos(df):
    # Calcular la media de 'age' para cada combinación de 'job', 'marital', 'education'
    age_mean = df.groupby(['job', 'marital', 'education'])['age'].mean().reset_index(name='mean_age')

    # Calcular la moda de 'marital' y 'education' para cada combinación
    marital_mode = df.groupby(['job', 'age', 'education'])['marital'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index(name='mode_marital')
    education_mode = df.groupby(['job', 'age', 'marital'])['education'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index(name='mode_education')

    # Imputar la media para 'age'
    for index, row in df[df['age'].isnull()].iterrows():
        mean_age = age_mean.loc[
            (age_mean['job'] == row['job']) &
            (age_mean['marital'] == row['marital']) &
            (age_mean['education'] == row['education']), 'mean_age'
        ]
        if not mean_age.empty:
            df.at[index, 'age'] = mean_age.values[0]

    # Imputar la moda para 'education'
    for index, row in df[df['education'].isnull()].iterrows():
        mode_education = education_mode.loc[
            (education_mode['job'] == row['job']) &
            (education_mode['age'] == row['age']) &
            (education_mode['marital'] == row['marital']), 'mode_education'
        ]
        if not mode_education.empty:
            df.at[index, 'education'] = mode_education.values[0]

    # Imputar la moda para 'marital'
    for index, row in df[df['marital'].isnull()].iterrows():
        mode_marital = marital_mode.loc[
            (marital_mode['job'] == row['job']) &
            (marital_mode['age'] == row['age']) &
            (marital_mode['education'] == row['education']), 'mode_marital'
        ]
        if not mode_marital.empty:
            df.at[index, 'marital'] = mode_marital.values[0]

    return df

# Cargar el dataset
bank = pd.read_csv('/content/drive/MyDrive/bank_dataset.csv')

# Eliminar outliers en las columnas especificadas
bank = remove_outliers(bank, 'age', factor=3)
bank = remove_outliers(bank, 'balance', factor=3)
bank = remove_outliers(bank, 'duration', factor=3)
bank = remove_outliers(bank, 'pdays', factor=3)
bank = remove_outliers(bank, 'previous', factor=3)

# Imputar valores nulos
bank = imputar_valores_nulos(bank)

# Calcular el número de suscripciones y no suscripciones en la variable 'deposit'
num_depositos = bank[bank['deposit'] == 'yes'].shape[0]
num_no_depositos = bank[bank['deposit'] == 'no'].shape[0]

# Calcular los porcentajes
total = num_depositos + num_no_depositos
percent_depositos = (num_depositos / total) * 100
percent_no_depositos = (num_no_depositos / total) * 100

# Mostrar los resultados en Streamlit
st.write(f"Porcentaje de depósitos: {percent_depositos:.2f}%")
st.write(f"Porcentaje de no depósitos: {percent_no_depositos:.2f}%")

# Crear gráfico de barras
fig, ax = plt.subplots()
ax.bar(["Depósitos (%d)" % num_depositos, "No Depósitos (%d)" % num_no_depositos],
       [num_depositos, num_no_depositos],
       color=["cyan", "red"],
       width=0.8)

ax.set_ylabel("Número de Personas")
ax.set_title("Distribución de Clientes Según la contratación de Depósitos")

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
