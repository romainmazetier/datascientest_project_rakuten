import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import gdown
import subprocess
import tempfile

# Chemin vers l'objet LabelEncoder sauvegardé
label_encoder_path = "../../models/label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)
class_labels = label_encoder.classes_


# Fonction pour télécharger un fichier depuis une URL Google Drive
def download_file_from_google_drive(file_id, output):
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(download_url, output, quiet=False)
    return output

# Fonction pour charger le CSV
@st.cache_data
def load_csv(file_id):
    csv_path = download_file_from_google_drive(file_id, "df_sample.csv")
    df = pd.read_csv(csv_path)
    return df

# Fonction pour décompresser et charger les images
@st.cache_data
def load_images(zip_file_id, image_paths):
    zip_path = download_file_from_google_drive(zip_file_id, "sample_img.zip")
    images = {}
    with zipfile.ZipFile(zip_path) as z:
        for file_name in image_paths:
            with z.open('sample_img/' + file_name) as file:
                img = Image.open(BytesIO(file.read()))
                images[file_name] = img
    return images

# IDs des fichiers CSV et ZIP partagés sur Google Drive
csv_file_id = '1wkr7ML3EBHPxM1i12_eaG8AMF_3J98s6'
zip_file_id = '1c9D_mLtaTYELX6Enyd5F_NMZ0j1Vyz1q'

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


# Charger le CSV et les images au lancement de l'application
if st.session_state.df is None:
    st.session_state.df = load_csv(csv_file_id)
if st.session_state.images is None:
    st.session_state.images = load_images(zip_file_id, st.session_state.df['img'].tolist())

# Fonction pour charger un produit aléatoire
def load_random_data():
    df = st.session_state.df
    images = st.session_state.images

    random_row = df.sample(n=1).iloc[0]
    text = random_row['designation']
    label = random_row['prdtypecode']
    image_path = random_row['img']
    image = images[image_path]

    st.session_state.text = text
    st.session_state.label = label
    st.session_state.image = image


def run_model_in_conda_env(conda_env, script_path, *args):
    command = ["conda", "run", "--name", conda_env, "python", script_path] + list(args)
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    return result.stdout.strip()


def display():

    st.header("Application pratique")

    # Section 1: Chargement du Texte et de l'Image
    st.subheader('1. Chargement du Produit', divider='rainbow')

    # Section pour charger les données aléatoires
    if st.button("Charger un produit test aléatoire"):
        load_random_data()

    # Afficher le texte et l'image sélectionnés
    if st.session_state.text and st.session_state.image:
        st.divider()
        st.write(st.session_state.text)
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(st.session_state.image, caption="Image du Produit")

        with col2:
            st.markdown("<p style='text-align: center;'>Classe à prédire : </p>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; border: 1px solid black; padding: 10px;'><span style='color: blue;'>{st.session_state.label}</span></div>", unsafe_allow_html=True)


    predictions = {}

    # Section 2: Choix du Modèle
    st.subheader('2. Choix du Modèle', divider='rainbow')

    # Section pour les modèles de texte
    st.markdown("**- Modèles de Texte**")
    text_model1 = st.checkbox("XLM-RoBERTa", key="text_model1")
    text_model2 = st.checkbox("LSTM", key="text_model2")
    text_model3 = st.checkbox("CNN (Benchmark)", key="text_model3")

    st.markdown("---")

    # Section pour les modèles d'images
    st.markdown("**- Modèles d'Images**")
    image_model1 = st.checkbox("ResNet50 (Benchmark)", key="image_model1")
    image_model2 = st.checkbox("EfficientNetB3", key="image_model2")

    st.markdown("---")

    # Section pour les modèles texte et images combinés
    st.markdown("**- Modèles Texte et Images Combinés**")
    combined_model1 = st.checkbox("Combined Embedding-VGG16", key="combined_model1")



    # Barre horizontale entre les sections
    st.markdown("---")


    # Section 3: Résultats
    st.subheader('3. Affichage des Prédictions', divider='rainbow')

    # Charger des données aléatoires
    if st.button(label='Submit'):

        # Sauvegarder l'image dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_image_path = temp_file.name
            st.session_state.image.save(temp_image_path)


        # Exécuter les modèles de texte
        if text_model1:
            text_result = run_model_in_conda_env("rakuten_project_torch", "xlm.py", st.session_state.text)
            predictions['XLM-RoBERTa'] = [float(x) for x in text_result.split()]

        if text_model2:
            text_result = run_model_in_conda_env("rakuten_project_tf", "LSTM.py", st.session_state.text)
            predictions['LSTM'] = [float(x) for x in text_result.split()]

        if text_model3:
            text_result = run_model_in_conda_env("rakuten_project_tf", "CNN_text.py", st.session_state.text)
            predictions['CNN (Benchmark)'] = [float(x) for x in text_result.split()]


        # Exécuter les modèles d'image
        if image_model1:
            image_result = run_model_in_conda_env("rakuten_project_tf", "resnet.py", temp_image_path)
            predictions['ResNet50 (Benchmark)'] = [float(x) for x in image_result.split()]

        if image_model2:
            image_result = run_model_in_conda_env("rakuten_project_tf", "efficientnet.py", temp_image_path)
            predictions['EfficientNetB3'] = [float(x) for x in image_result.split()]

        # Exécuter les modèles combinés
        if combined_model1:

            combined_result = run_model_in_conda_env("rakuten_project_tf", "combi.py", st.session_state.text, temp_image_path)
            predictions['Combined Embedding-VGG16'] = [float(x) for x in combined_result.split()]



        # Afficher les résultats
        st.write("Résultats des prédictions :")
        labels = label_encoder.classes_
        
        # Créer un DataFrame pour les probabilités
        data = []
        for model_name, probabilities in predictions.items():
            for class_idx, probability in enumerate(probabilities):
                data.append((model_name, class_labels[class_idx], probability))

        df_probs = pd.DataFrame(data, columns=["Modèle", "Classe", "Probabilité"])

        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 8))

        # Paramètres pour les barres
        n_classes = len(class_labels)
        n_models = len(predictions)
        bar_width = 0.8 / n_models  # Largeur des barres

        # Position des barres sur l'axe x
        x = np.arange(n_classes)

        # Créer un barplot pour chaque modèle
        for i, (model_name, probabilities) in enumerate(predictions.items()):
            bar_positions = x + (i * bar_width) - bar_width * (n_models - 1) / 2
            ax.bar(bar_positions, probabilities, width=bar_width, label=model_name)

        # Configuration de l'axe x
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_xlabel("Classe")
        ax.set_ylabel("Probabilité")
        ax.set_title("Probabilités des classes par modèle")
        ax.legend()

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)

