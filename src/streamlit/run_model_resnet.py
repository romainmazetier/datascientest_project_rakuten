import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.applications import ResNet50
import tensorflow_addons as tfa
from sklearn.metrics import classification_report

import joblib
import h5py
import numpy as np

# Chemin vers l'objet LabelEncoder sauvegardé
label_encoder_path = "../../models/label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)
class_labels = label_encoder.classes_

def run_model_combi(X_test_img, y_test):
    model = load_model('../../models/ResNet50.h5')
    pred = model.predict(X_test_img)
    
    y_pred = np.argmax(pred, axis = -1)
    y_true = np.argmax(y_test, axis = -1)
    
    report = classification_report(np.array(y_true), np.array(y_pred), target_names=list(map(str,label_encoder.classes_)), output_dict=True)
    cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return report, normalized_cm


if __name__ == "__main__":

    
    # Lire le tableau à partir du fichier HDF5
    with h5py.File('xy_val_224.h5', 'r') as f:
        x_img_val = f['x_img_val_cropped'][:]
        y_val = f['y_val'][:]

    report, cm = run_model_combi(x_img_val, y_val)

    # Sauvegarder les résultats pour les lire dans Streamlit
    np.save('npy/report.npy', report)
    np.save('npy/cm.npy', cm)
