import sys
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import h5py
import tensorflow as tf
import tensorflow_addons as tfa
import joblib

# Chemin vers l'objet LabelEncoder sauvegardé
label_encoder_path = "../../models/label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)
class_labels = label_encoder.classes_

def run_model_combi(X_test_txt, X_test_img, y_test):
    model = load_model('../../models/combi.h5')
    pred = model.predict([X_test_img, X_test_txt])
    
    y_pred = np.argmax(pred, axis = -1)
    y_true = np.argmax(y_test, axis = -1)
    
    report = classification_report(np.array(y_true), np.array(y_pred), target_names=list(map(str,label_encoder.classes_)), output_dict=True)
    cm = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return report, normalized_cm


if __name__ == "__main__":

    
    # Lire le tableau à partir du fichier HDF5
    with h5py.File('xy_val_100.h5', 'r') as f:
        test_sequences_matrix = f['test_sequences_matrix'][:]
        x_img_val = f['x_img_val'][:]
        y_val = f['y_val'][:]


    # Exemple de données de test
    X_test_txt = test_sequences_matrix
    X_test_img = x_img_val
    y_test = y_val
    
    report, cm = run_model_combi(X_test_txt, X_test_img, y_test)
    
    # Sauvegarder les résultats pour les lire dans Streamlit
    np.save('npy/report.npy', report)
    np.save('npy/cm.npy', cm)