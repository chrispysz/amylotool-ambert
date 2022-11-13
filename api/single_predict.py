from transformers import BertForSequenceClassification, BertTokenizer
import torch
import logging
import time


def predict_single(seq):

    threshold = 0.5

    model = BertForSequenceClassification.from_pretrained(
        "./models/AmBERT")

    tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/")

    inputs = tokenizer(" ".join(seq), return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.softmax(logits, axis=-1).detach().numpy()
    seq_dict = {
        "sequence": seq,
        "prediction": str(prediction[0][1])
    }

    if prediction[0][1] > threshold:

        return ("Positive", seq_dict)
    else:

        return ("Negative", seq_dict)
