import streamlit as st
from PIL import Image

img = Image.open("../../data/photo/i.jpg")

st.title('Web-приложение для вывода моделей ML и анализа данных')

st.title("Обо мне")
st.header("ФИО")
st.write("Тютюник Мария Сергеевна")

st.header("Группа")
st.write("МО-221")

st.header("Фото")
st.image(img, caption = 'Фото разработчика', width = 400)
