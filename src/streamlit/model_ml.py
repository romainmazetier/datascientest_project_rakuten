import streamlit as st
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_option('deprecation.showPyplotGlobalUse', False)

# Chemin vers l'objet LabelEncoder sauvegardé
label_encoder_path = "../../models/label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)
class_labels = label_encoder.classes_

# Fonction pour exécuter un script Python dans un environnement Conda
def run_script_in_conda_env(env_name, script_name):
    process = subprocess.run(
        f"conda run -n {env_name} python {script_name}",
        shell=True,
        check=True
    )
    return process

# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Dictionnaire des modèles et de leurs descriptions
model_descriptions = {
    'Combiné : Random Forest': {
        'text': """

            Input : Texte + Images

            Le modèle Random Forest crée une collection d’arbres de décision (dans notre cas, 500). 
            
            Chaque arbre a une liste d’attributs choisis au hasard qui lui permet de donner une classification temporaire à un élément. 
            
            On fait ensuite un vote majoritaire pour obtenir la classification finale de chaque élément.

        """,
        'image': "image/rf.png",
        'col_size': 2
    },
    'Texte : Random Forest': {
        'text': """

            Input : Texte

            Le modèle Random Forest crée une collection d’arbres de décision (dans notre cas, 500). 
            
            Chaque arbre a une liste d’attributs choisis au hasard qui lui permet de donner une classification temporaire à un élément. 
            
            On fait ensuite un vote majoritaire pour obtenir la classification finale de chaque élément.

        """,
        'image': "image/rf.png",
        'col_size': 2
    },
    'Texte : SGD Classifier': {
        'text': """

            Input : Texte

            Le modèle SGD, pour Stochastic Gradient Descent, calcule une frontière entre deux classes. 
            
            Dans le cas d’une classification multi-classes comme notre projet, il calcule une frontière entre chaque classe et les 26 autres (One versus All). 
            
            Chaque élément est ensuite classé en fonction de sa distance aux 27 frontières. 
            
            Il est à noter que ce modèle est extrêmement rapide, son entraînement sur les données texte ne prend que quelques secondes.

        """,
        'image': "image/sgd.png",
        'col_size': 2
    },
    'Image : Random Forest + UMAP': {
        'text': """

            Input : Images 

            Dans ce modèle et dans le modèle Combinés : Random Forest, nous avons réduit la dimension de nos images avec la transformation UMAP (pour Uniform Manifold Approximation and Projection). 
            
            Les résultats sont cependant peu satisfaisants, c’est pourquoi nous avons essayé d’autres méthodes sur les images.

        """,
        'image': "image/umap.png",
        'col_size': 0.75
    },
    'Image : Random Forest + HOG': {
        'text': """

            Input : Images 

            Le filtre HOG, pour Histogram of Oriented Gradients, réduit la dimension d’une image. 
            
            L’algorithme divise l’image en blocs de pixels et calcule un gradient par bloc. Dans notre exemple, on passe de 50176 pixels à 8100 hog features.

            Nous avons ensuite passé ces images filtrées dans un modèle Random Forest.

        """,
        'image': "image/hog.png",
        'col_size': 1
    },
    # Ajoutez des descriptions pour d'autres modèles si nécessaire
}


def display():

    # Interface utilisateur de Streamlit
    st.header("Modèles de Machine Learning")

    st.subheader("1. Les modèles", divider='rainbow')

    # Dictionnaire des modèles et des environnements correspondants
    models = {
        'Combiné : Random Forest':      ('rakuten_project_tf', 'run_model_rf_combi.py'),
        'Texte : Random Forest':        ('rakuten_project_tf', 'run_model_rf.py'),
        'Texte : SGD Classifier':       ('rakuten_project_tf', 'run_model_sgd.py'),
        'Image : Random Forest + UMAP': ('rakuten_project_tf', 'run_model_umap.py'),
        'Image : Random Forest + HOG':  ('rakuten_project_tf', 'run_model_hog.py')
    }

    # Sélecteur de modèle
    model_name = st.selectbox("Sélectionnez un modèle", list(models.keys()))

    model_info = model_descriptions[model_name]
    col1, col2 = st.columns([model_info['col_size'], 1])
    with col1:
        st.markdown(model_info['text'])

    with col2:
        st.image(model_info['image'], caption=f"Image de {model_name}")

    st.subheader("2. Les résultats", divider='rainbow')

    if st.button("Afficher les résultats"):
        env_name, script_name = models[model_name]
        
        # Exécuter le script dans l'environnement Conda approprié
        try:
            run_script_in_conda_env(env_name, script_name)
            
            # Charger les résultats sauvegardés par les scripts
            report = np.load('npy/report.npy', allow_pickle=True).item()
            accuracy = report['accuracy']
            weighted_f1_score = report['weighted avg']['f1-score']
            cm = np.load('npy/cm.npy')
            
            report_df = pd.DataFrame(report).transpose()
            st.subheader("Évaluation")
            st.markdown(f"Accuracy: **<span style='color:blue;'>{accuracy:.2f}</span>**", unsafe_allow_html=True)
            st.markdown(f"Weighted F1-score: **<span style='color:blue;'>{weighted_f1_score:.2f}</span>**", unsafe_allow_html=True)


            st.subheader("Rapport de classification")
            # Affichage du tableau avec st.table
            st.table(report_df)
            
            st.subheader("Matrice de confusion")

            # Création de la figure
            plt.figure(figsize=(15, 8))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', 
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

            plt.title('Matrice de confusion normalisée')
            plt.xlabel('Prédictions')
            plt.ylabel('Valeurs réelles')

            # Rotation des étiquettes des axes pour éviter les chevauchements
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)

            plt.tight_layout()
            st.pyplot()
        
        except subprocess.CalledProcessError as e:
            st.error(f"Erreur lors de l'exécution du script : {e}")

