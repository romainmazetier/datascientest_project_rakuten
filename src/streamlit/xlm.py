# xlm.py

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import joblib
import sys

def load_model_and_tokenizer():
    model_path = "../../models/transformers/"
    tokenizer_path = "../../models/transformers/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path).to(device)

    return tokenizer, model

def load_label_encoder():
    label_encoder_path = "../../models/label_encoder.pkl"
    label_encoder = joblib.load(label_encoder_path)
    return label_encoder

def classify_text(text_input):
    tokenizer, model = load_model_and_tokenizer()

    sequences = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in sequences.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits

    probabilities = torch.softmax(prediction, dim=1).cpu().numpy()[0]

    return probabilities

if __name__ == "__main__":
    text_input = sys.argv[1]
    probabilities = classify_text(text_input)
    probabilities_str = " ".join([str(prob) for prob in probabilities])
    print(probabilities_str)


