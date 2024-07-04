import streamlit as st
import pandas as pd
from streamlit_image_comparison import image_comparison

def display():


    st.header("Préparation des Données")
    st.subheader('1. Nettoyage du texte', divider='rainbow')
    st.write("Nous allons expliquer comment nous avons nettoyé et préparé les données, d'abord pour le texte ensuite pour les images :")
    pd.set_option('display.max_colwidth', None)
    prepro = pd.read_csv("datasets/preprocessing.csv",encoding='utf-8')
    st.markdown(prepro[['Text', 'Methode']].style.hide(axis="index").to_html(), unsafe_allow_html=True)
    st.subheader('2. Nettoyage des images', divider='rainbow')
    image_comparison(img1="image/image_457047496_product_50418756.jpg",
    img2="image/image_457047496_product_50418756_resized.jpg",
    label1="Before",
    label2="After",
    width=700,
    starting_position=50,
    show_labels=True,
    make_responsive=True,
    in_memory=True)