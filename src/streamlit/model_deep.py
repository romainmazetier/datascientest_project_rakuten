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
    'Image : ResNet50 (Benchmark)': {
        'text': """

            Input : Image

            ResNet50 est un des premiers modèles de réseau de neurones convolutif profond. 
            
            En raison de son succès sur des tâches de classification d'images à grande échelle comme ImageNet, ResNet50 est souvent utilisé comme benchmark dans le transfer learning, c'était le modèle de benchmark utilisé par le challenge rakuten https://challengedata.ens.fr/.

        """,
        'image': "image/resnet.png",
        'col_size': 0.5
    },

    'Image : EfficientNetB3': {
        'text': """

            Input : Image

            Pour améliorer nos résultats nous avons décidé d'approfondir notre approche par transfer learning, en prenant cette fois-ci le modèle EfficientNet pré-entraîné sur la base d'images "ImageNet". 
            
            Nous avons supprimé les dernières couches pour les remplacer par des couches personnalisées puis avons entraîné celui-ci en deux étapes distinctes. 
            
            L'objectif était d'améliorer notre benchmark avec ce modèle qui a parmi les meilleurs scores dans les modèles pré-entrainés (https://keras.io/api/applications/)


        """,
        'image': "image/efficientnet.png",
        'col_size': 0.5
    },

    'Texte : LSTM': {
        'text': """

            Input : Texte

            Technologie : RNN

            Les architectures récurrentes telles que les LSTM ont constitué l'état de l'art en matière de NLP. 
            
            Ces architectures contiennent une boucle de rétroaction dans les connexions du réseau qui permet à l'information de se propager d'une étape à l'autre, ce qui les rend idéales pour modéliser des données séquentielles telles que des textes.

            Voici l’architecture très simple que nous avons proposée : 

        """,
        'image': "image/LSTM.png",
        'col_size': 4
    },

    'Texte : CNN (Benchmark)': {
        'text': """

            Input : Texte

            Technologie : CNN

            Dans le cadre du Challenge Rakuten, l’équipe chargée de proposer le challenge a mis à disposition un modèle “benchmark”.

            Le réseau utilise plusieurs convolutions parallèles avec différentes tailles de filtres pour capturer des caractéristiques locales de différentes tailles.

            Voici l’architecture plus complexe : 

        """,
        'image': "image/cnn_text.png",
        'col_size': 0.5
    },

    'Texte : XLM-RoBERTa': {
        'text': """

            Input : Texte

            Technologie : Transformers

            Les Transformers sont une classe de modèles d'apprentissage profond qui se sont révélés efficaces dans une large gamme de tâches de NLP et de CV en utilisant une architecture basée sur l'attention pour capturer les dépendances à longue distance entre les éléments de séquence. 
            
            Ils fonctionnent en traitant simultanément l'ensemble de la séquence d'entrée plutôt que de manière séquentielle, ce qui permet des performances élevées sur diverses tâches sans nécessiter de prétraitement spécifique à la tâche. 
            
            Ils se révèlent notamment très performants pour la classification de textes.

            Nous utiliserons XLM-RoBERTa qui est une variante multilingue de RoBERTa, optimisée pour l'apprentissage multilingue en utilisant des techniques d'entraînement supplémentaires et une architecture de Transformer robuste.

        """,
        'image': "image/xlm.png",
        'col_size': 1
    },

    'Modèle Bimodal': {
        'text': """

            Input : Texte + Images 

            Technologie : Embedding + CNN

            Nous avons construit un modèle capable de traiter à la fois des données d'image et de texte en entrée en utilisant des réseaux de neurones distincts pour chaque type de données. 
            
            Ces réseaux de neurones sont ensuite fusionnés à la fin du processus pour combiner les représentations extraites des deux modalités, permettant ainsi au modèle de capturer des relations complexes entre l'image et le texte. 
            
            Ici, il s'agit d'un modèle de simple embedding pour les données textuelles et un VGG16 très basique pour les images.

            Voici l’architecture très simple que nous avons proposée : 

        """,
        'image': "image/combi.png",
        'col_size': 2
    },
    # Ajoutez des descriptions pour d'autres modèles si nécessaire
}




def display():

    # Interface utilisateur de Streamlit
    st.header("Modèles de Deep Learning")

    st.subheader("1. Les modèles", divider='rainbow')

    # Dictionnaire des modèles et des environnements correspondants
    models = {
        'Image : ResNet50 (Benchmark)': ('rakuten_project_tf', 'run_model_resnet.py'),
        'Image : EfficientNetB3': ('rakuten_project_tf', 'run_model_effi3b.py'),
        'Texte : LSTM': ('rakuten_project_tf', 'run_model_lstm.py'),
        'Texte : CNN (Benchmark)': ('rakuten_project_tf', 'run_model_cnn.py'),
        'Texte : XLM-RoBERTa': ('rakuten_project_torch', 'run_model_xlm.py'),
        'Modèle Bimodal': ('rakuten_project_tf', 'run_model_combi.py'),
    }

    # Sélecteur de modèle
    model_name = st.selectbox("Sélectionnez un modèle", list(models.keys()))

    model_info = model_descriptions[model_name]
    col1, col2 = st.columns([model_info['col_size'], 1])
    with col1:
        st.markdown(model_info['text'])

    with col2:
        st.image(model_info['image'], caption=f"{model_name}")

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

