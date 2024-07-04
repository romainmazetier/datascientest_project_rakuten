import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib
import tensorflow_addons as tfa
import sys
import tensorflow as tf
import logging

# Configurer TensorFlow pour désactiver les sorties de journalisation
tf.get_logger().setLevel(logging.ERROR)

# Charger le modèle entraîné
model_path = '../../models/LSTM.h5'  # Remplacer par le nom du fichier modèle approprié
model = load_model(model_path)

# Charger le Tokenizer
with open('../../models/tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

# Charger le LabelEncoder
label_encoder = joblib.load("../../models/label_encoder.pkl")


# Paramètres du tokenizer
max_len = 34

# Fonction de classification
def classify_text(text):
    
    sequence = tok.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    probabilities = model.predict(padded_sequence, verbose=0)
    return probabilities[0]

# Demander une entrée utilisateur et afficher les probabilités par classe
if __name__ == "__main__":   
    text_input = sys.argv[1]
    probabilities = classify_text(text_input)
    probabilities_str = " ".join([str(prob) for prob in probabilities])
    print(probabilities_str)
    