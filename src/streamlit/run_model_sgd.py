import h5py
import numpy as np

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

import joblib


def run_model(X, y_val):

    # Charger le modèle
    model = joblib.load('../../models/sgd_txt.pkl')
    
    # Utiliser le modèle chargé pour faire des prédictions
    y_pred = model.predict(X)

    labels = np.unique(y_val)
    
    report = classification_report(np.array(y_val), np.array(y_pred), output_dict=True)
    
    cm = confusion_matrix(y_val, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return  report, cm_normalized

if __name__ == "__main__":

    # Charger un tableau NumPy à partir du fichier HDF5
    with h5py.File('val_ml_data.h5', 'r') as hf:
        X_val = hf['X_val_txt_reduced'][:]
        y_val = hf['y_val'][:]

    report, cm = run_model(X_val, y_val)
    
    # Sauvegarder les résultats pour les lire dans Streamlit
    np.save('npy/report.npy', report)
    np.save('npy/cm.npy', cm)