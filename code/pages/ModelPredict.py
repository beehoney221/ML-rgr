import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import lightgbm
import keras
import pickle


def LearnWithTeacher(arr):
    
    st.title("Обучение с учителем")

    models = ["PolynomialRegression", "DecisionTreeRegressor"]
    models_type = st.selectbox("Выберите модель", models)

    if models_type is not None:
        if models_type == "DecisionTreeRegressor":
            st.header("DecisionTreeRegressor")
            with open('data/model/DecisionTreeRegressor.pkl', 'rb') as f:
                tree_model = pickle.load(f)
            Pred(tree_model, arr)

        elif models_type == "PolynomialRegression":
            st.header("PolynomialRegression")
            st.subheader("ElasticNet")
            with open('data/model/PolyRegressionEN.pkl', 'rb') as file:
                poly_model = pickle.load(file)
            Pred(poly_model, arr)

def Ensembles(arr):

    st.title("Ансамбли")

    models = ["BaggingRegressor", "GradientBoostingRegressor"]
    models_type = st.selectbox("Выберите модель", models)

    if models_type is not None:
        if models_type == "BaggingRegressor":
            st.header("BaggingRegressor")
            with open('data/model/BaggingRegressor.pkl', 'rb') as file:
                bag_model = pickle.load(file)
            Pred(bag_model, arr)

        elif models_type == "GradientBoostingRegressor":
            st.header("GradientBoostingRegressor")
            with open('data/model/GradientBoostingRegressor.pkl', 'rb') as file:
                grad_model = pickle.load(file)
            Pred(grad_model, arr)

def AdvancedEnsembles(arr):

    st.title("Продвинутые ансамбли")

    models = ["LightGBM (Light Gradient Boosted Machine)", "XGBoost (eXtreme Gradient Boosting)"]
    models_type = st.selectbox("Выберите модель", models)

    if models_type is not None:
        if models_type == "LightGBM (Light Gradient Boosted Machine)":
            st.header("LGBMRegressor")
            lgbm_model = lightgbm.Booster(model_file = 'data/model/LGBMRegressor.txt')
            Pred(lgbm_model, arr)

        elif models_type == "XGBoost (eXtreme Gradient Boosting)":
            st.header("XGBRegressor")
            xgb_model = XGBRegressor()
            xgb_model.load_model('data/model/XGBRegressor.json')
            Pred(xgb_model, arr)

def DNN(arr):

    st.title("Нейронные сети")

    st.header("DNN")
    dnn_model = keras.models.load_model('data/model/RegressionDNN.keras')
    Pred(dnn_model, arr)

def Pred(model, arr):

    pred = model.predict(arr)
    pred = pred.flatten()
    if (len(pred) > 1):
        st.write(f"Предсказанные значения")
        pred_df = pd.DataFrame(data = pred, columns=["price_usd"])
        col1, col2, col3 = st.columns(3)
        with col2:
            st.write(round(pred_df, 4))
    else:
        if(len(pred) == 1):
            st.write(f"Предсказанная стоимость автомобиля - ${round(pred[0], 2)}")
        else:
            st.write(f"Не удалось предсказать")


def PredictObject():

    st.header("Ввод данных")

    bin = {'есть' : 0,
       'отстутствует' : 1}
    
    trans = {'автомат' : 0, 
         'механика' : 1}
    transmission = st.selectbox("Коробка передач", trans.keys())

    col = {'черный' : 0,
        'синий' : 1,
        'коричневый' : 2,
        'зеленый' : 3,
        'серый' : 4,
        'оранжевый' : 5,
        'красный' : 7,
        'серебристый' : 8,
        'фиолетовый' : 9,
        'белый' : 10,
        'желтый' : 11,
        'другой' : 6}
    color = st.selectbox("Цвет автомобиля", col.keys())

    odometer_value = st.slider('Значение одометра', 0, 600000, step = 1000)

    fuel = {'дизель' : 0,
            'электричество' : 1,
            'газ' : 2,
            'бензин' : 3,
            'гибрид-дизель' : 4,
            'гибрид-бензин' : 5}
    engine_fuel = st.selectbox("Топливо двигателя", fuel.keys())

    engine_has_gas = st.selectbox("Наличие бензин", bin.keys())

    eType = {'дизель' : 0,
            'электричество' : 1,
            'бензин' : 2}
    engine_type = st.selectbox("Тип двигателя", eType.keys())

    engine_capacity = st.number_input("Объем двигателя", min_value=0.5, max_value=3.5, step = 0.1)

    bType = {'кабриолет' : 0,
            'купе' : 1,
            'хэтчбек' : 2,
            'лифтбек' : 3,
            'лимузин' : 4,
            'микроавтобус' : 5,
            'минивэн' : 6,
            'пикап' : 7,
            'седан' : 8,
            'внедорожник' : 9,
            'универсал' : 10,
            'фургон' : 11}
    body_type = st.selectbox("Тип кузова", bType.keys())

    has_warranty = st.selectbox("Наличие гарантии", bin.keys())

    stat = {'запасной' : 0,
            'новый' : 1,
            'старый' : 2}
    state = st.selectbox("Состояние", stat.keys())

    drTrain = {'вся' : 0,
            'передняя' : 1,
            'задняя' : 2}
    drivetrain = st.selectbox("Трансмиссия", drTrain.keys())

    is_exchangeable = st.selectbox("Наличие контрактной детали", bin.keys())

    region = {'Брестская обл.' : 0,
            'Витебская обл.' : 1,
            'Гомельская обл.' : 2,
            'Гродненская обл.' : 3,
            'Минская обл.' : 4,
            'Могилевская обл.' : 5}
    location_region = st.selectbox("Регион местоположения", region.keys())

    number_of_photos = st.number_input("Количество фото", min_value=1, step = 1)

    up_counter = st.slider("Количество раз, когда автомобиль поднимался", min_value=0, max_value=200, step=1)

    duration_listed = st.slider("Количество дней в списке (до продажи)", min_value=0, max_value=2500, step=1)

    age = st.number_input("Возраст автомобиля", min_value=0, max_value=150)


    arr = np.array([trans[transmission], col[color], odometer_value, fuel[engine_fuel], bin[engine_has_gas], eType[engine_type], 
                    engine_capacity, bType[body_type], bin[has_warranty], stat[state], drTrain[drivetrain], bin[is_exchangeable], region[location_region],
                    number_of_photos, up_counter, duration_listed, age])

    check_box = st.checkbox("Выбрать модель")

    if check_box:

        arr = arr.reshape(1, -1)
        st.write(np.array([transmission, color, odometer_value, engine_fuel, engine_has_gas, engine_type, 
                    engine_capacity, body_type, has_warranty, state, drivetrain, is_exchangeable, location_region,
                    number_of_photos, up_counter, duration_listed, age]).reshape(1, -1))

        models = ["Обучение с учителем", "Ансамбли", "Продвинутые ансамбли", "Нейронные сети"]

        models_type = st.selectbox("Тип модели", models)

        if models_type is not None:
            if models_type == "Обучение с учителем":
                LearnWithTeacher(arr)
            elif models_type == "Ансамбли":
                Ensembles(arr)    
            elif models_type == "Продвинутые ансамбли":
                AdvancedEnsembles(arr)
            elif models_type == "Нейронные сети":
                DNN(arr)

def PredictDataset():
    
    st.header("Загрузка датасета")
    load_df = st.file_uploader("Загрузите файл типа .csv", type="csv")

    if load_df is not None:
        df = pd.read_csv(load_df)
    
        df  = df[['transmission', 'color', 'odometer_value', 'engine_fuel', 'engine_has_gas', 'engine_type', 
                'engine_capacity', 'body_type', 'has_warranty', 'state', 'drivetrain', 'is_exchangeable', 'location_region',
                'number_of_photos', 'up_counter', 'duration_listed', 'age']]
        st.write(df[:5])
        df_sc = sc.transform(df)
        check_box = st.checkbox("Выбрать модель")
    
        if check_box:
            models = ["Обучение с учителем", "Ансамбли", "Продвинутые ансамбли", "Нейронные сети"]
            models_type = st.selectbox("Тип модели", models)

            if models_type is not None:
                if models_type == "Обучение с учителем":
                    LearnWithTeacher(df_sc)
                elif models_type == "Ансамбли":
                    Ensembles(df_sc)    
                elif models_type == "Продвинутые ансамбли":
                    AdvancedEnsembles(df_sc)
                elif models_type == "Нейронные сети":
                    DNN(df_sc)



st.title("Работа с моделями регрессии")

st.header("Выберите тип предсказания")
types_of_predict = ["Предсказание для одного объекта", "Предсказание для датасета"]

pred_type = st.selectbox("Тип предсказаний", types_of_predict)

with open('data/model/StandardScaler.pkl', 'rb') as file:
    sc = pickle.load(file)

if pred_type is not None:
    if pred_type == "Предсказание для одного объекта":
        PredictObject()
    elif pred_type == "Предсказание для датасета":
        PredictDataset() 
