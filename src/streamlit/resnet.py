import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.applications import ResNet50
import tensorflow_addons as tfa

import joblib
import numpy as np
import cv2
import sys

# Taille image
width = 224
height = 224

# Chemin vers l'objet LabelEncoder sauvegardé
label_encoder_path = "../../models/label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)
class_labels = label_encoder.classes_

def run_classify(X_test_img):
    model = load_model('../../models/ResNet50.h5')
    probabilities = model.predict(X_test_img, verbose=0)
    return probabilities[0]
    

# Demander une entrée utilisateur et afficher les probabilités par classe
if __name__ == "__main__":   

    img = cv2.imread(sys.argv[1])
    resized_img = cv2.resize(img, (width, height))
    X_img = np.reshape(resized_img, (1,width,height,3))

    probabilities = run_classify(X_img)
    probabilities_str = " ".join([str(prob) for prob in probabilities])
    print(probabilities_str)