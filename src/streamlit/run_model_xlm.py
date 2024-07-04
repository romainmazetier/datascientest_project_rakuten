# run_model_xlm.py

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import joblib
import sys
import h5py
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def run_model_xlm(loaded_dataset_dict, y_test):

    label_encoder_path = "../../models/label_encoder.pkl"
    label_encoder = joblib.load(label_encoder_path)
    
    model_path = "../../models/transformers/"
    tokenizer_path = "../../models/transformers/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path).to(device)

    num_labels = 27
    model_ckpt = 'xlm-roberta-base'
    
    batch_size = 32
    logging_steps = max(1, len(loaded_dataset_dict) // batch_size)
    model_name = f"{model_ckpt}-finetuned-raw"
    training_args = TrainingArguments(output_dir=model_name,
                    num_train_epochs=5,
                    learning_rate=2e-5,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    weight_decay=0.01,
                    evaluation_strategy="epoch",
                    disable_tqdm=False,
                    logging_steps=logging_steps,
                    push_to_hub=True,
                    log_level="error")


    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=loaded_dataset_dict,
        tokenizer=tokenizer
    )

    preds_output = trainer.predict(loaded_dataset_dict)
    y_preds = np.argmax(preds_output.predictions, axis=1)

    report = classification_report(np.array(y_test), np.array(y_preds), target_names=list(map(str,label_encoder.classes_)), output_dict=True)
    normalized_cm = confusion_matrix(y_test, y_preds, normalize="true")

    return report, normalized_cm



if __name__ == "__main__":

    # Chargement du dataset
    loaded_dataset_dict = torch.load('dataset_encoded_test.pth')
    y_test = np.array(loaded_dataset_dict["label"])

    report, normalized_cm = run_model_xlm(loaded_dataset_dict, y_test)

    # Sauvegarder les résultats pour les lire dans Streamlit
    np.save('npy/report.npy', report)
    np.save('npy/cm.npy', normalized_cm)
