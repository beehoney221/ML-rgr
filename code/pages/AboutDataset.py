import streamlit as st
import pandas as pd

data_start = pd.read_csv('data/dataset/cars.csv')
data_prepared = pd.read_csv('data/dataset/cars_prepared.csv')
data_prepared = data_prepared.drop(["Unnamed: 0"], axis=1)

st.title('О наборе данных')

st.header('Тематика')
st.markdown("""
            Набор данных содержит автомобильные объявления с множеством категориальных и числовых характеристик. 
            Данные собирались из различных веб-ресурсов с целью изучить рынок подержанных автомобилей и попытаться
            построить модель, которая эффективно прогнозирует цену автомобиля на основе его параметров.
            """)

st.header('Описание полей')
st.markdown("""
            - `manufacturer_name` — имя производителя
            - `model_name` — название модели
            - `transmission` — коробка передач
            - `color` — цвет
            - `odometer_value` — значение одометра (пройденный путь)
            - `year_produced` — год производства
            - `engine_fuel` — топливо двигателя
            - `engine_has_gas` — наличие бензина
            - `engine_type` — тип двигателя
            - `engine_capacity` — объем двигателя
            - `body_type` — тип кузова
            - `has_warranty` — наличие гарантии
            - `state` — состояние
            - `drivetrain` — трансмиссия
            - `price_usd` — цена в долларах
            - `is_exchangeable` — контрактная деталь (взаимозаменяемы ли детали машины)
            - `location_region` — регион местоположения
            - `number_of_photos` — количество фотографий
            - `feature_0` - `feature_9` — особенность 0-9
            - `up_counter` — количество раз, когда автомобиль поднимался
            - `duration_listed` — количество дней, в течение которых он был указан в списке

            Ниже представлен ещё не обработанный датасет
            """)
st.write(data_start[:5])

st.header('Особенности предобработки данных')
st.markdown("""
            - В ходе обсуждения было решено удалить столбцы `feature`, так как не понятно, 
            какую информацию они содержат, и повлияет ли она на работу модели 
            - Пропущенные значения в столбце `engine_capacity` были заменены модой, значение которой совпадало со средним и медианой
            - Изменён тип столбца `year_produced` на тип *date*
            - Удалены явные дубликаты
            - Преобразованы категориальные переменные
            - Добавление признака `age` - возраст автомобиля, удаление признака `year_produced`
            - Удалены выбросы в столбцах: `odometer_value`, `engine_capacity`, `price_usd`, 
            `number_of_photos`, `duration_listed` с помощью метода межквартильных размахов.
            
            В результате предобработки были заполнены пропущенные значения, заменены типы 
            некоторых столбцов, обогащены данные, преобразованы категориальные переменные, удалены лишние признаки, явные дубликаты, выбросы.
            
            Ниже представлен уже обработанный датасет
            """)
st.write(data_prepared[:5])
