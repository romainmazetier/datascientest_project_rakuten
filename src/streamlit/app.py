import streamlit as st
from application import display as display_application
from model_ml import display as display_model_ml
from model_deep import display as display_model_deep
from dataset import display as display_dataset
from data_prep import display as display_dataprep
from intro_conclu import display_intro, display_conclu



# Initialisation des clés dans st.session_state pour éviter les KeyError
if 'text' not in st.session_state:
    st.session_state.text = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'images' not in st.session_state:
    st.session_state.images = None
if 'label' not in st.session_state:
    st.session_state.label = None
if 'df' not in st.session_state:
    st.session_state.df = None


# Définition des options de navigation
pages = {
        "Introduction au Projet": display_intro, 
        "Le Jeu de Données": display_dataset, 
        "Préparation des Données": display_dataprep,
        "Modélisation Machine Learning": display_model_ml,
        "Modélisation Deep Learning": display_model_deep,
        "Application Pratique": display_application,
        "Conclusion": display_conclu
}

# Menu de sélection de page dans la sidebar
selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# Exécute la fonction associée à la page sélectionnée
pages[selected_page]()


