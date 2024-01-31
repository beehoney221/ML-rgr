import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def HeatMap():
    st.title("Тепловая карта")
    plt.subplots(figsize=(15, 8))
    sns.heatmap(data.corr(), cmap="Blues", annot=True, fmt=".2f", linewidth=.5)
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)

def BoxPlot():
    st.title("Ящик с усами")    
    features = ["odometer_value", "engine_capacity", "price_usd", "number_of_photos" , "duration_listed"]
    ft = st.selectbox("Выберите признак", features)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x = data[ft])
    plt.title(f'Диаграмма "ящик с усами" для признака {ft}')
    plt.xlabel('Значение')
    st.pyplot(plt)

def Hist():
    st.title("Гистограмма")    
    features = ["odometer_value", "engine_capacity", "price_usd", "number_of_photos" , "duration_listed"]
    ft = st.selectbox("Выберите признак", features, key='selectbox1')
    plt.figure(figsize=(10, 6))
    sns.histplot(data[ft], bins=100)
    plt.title(f'Гистограмма для признака {ft}')
    st.pyplot(plt)

def LmlPlot():
    st.title("Диаграмма рассеивания для первой 1000 строк")    
    features = ["odometer_value", "engine_capacity", "price_usd", "number_of_photos" , "duration_listed"]
    ft1 = st.selectbox("Выберите первый признак", features, key='selectbox1')
    ft2 = st.selectbox("Выберите второй признак", features, key='selectbox2')
    plt.figure(figsize=(10, 6))
    plt.scatter (data[ft1][:1000], data[ft2][:1000], s= 60 , c='purple')
    plt.title(f'Диаграмма рассеивания для признаков {ft1} и {ft2}')
    st.pyplot(plt)


data = pd.read_csv('data/dataset/cars_prepared.csv')
data = data.drop(['Unnamed: 0', 'manufacturer_name', 'model_name'], axis=1)

st.title('Визуализация')

st.write("Из обработанного датасета были удалены категориальные переменные, которые не удалось преобрзовать. Ниже представлен датасет, данные которого мы будем визуализировать.")
st.write(data[:5])

st.write("Выберите тип визуализации, который Вас интересует:")

vis_types = ['Тепловая карта', 'Ящик с усами', 'Гистограмма', 'Диаграмма рассеивания']

vis_type = st.selectbox("Тип визуализации", vis_types)

if vis_type is not None:
    if vis_type == "Тепловая карта":
        HeatMap()
    elif vis_type == "Ящик с усами":
        BoxPlot()    
    elif vis_type == "Гистограмма":
        Hist()
    elif vis_type == "Диаграмма рассеивания":
        LmlPlot()
