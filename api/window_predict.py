import torch
import logging
import time
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("./models/AmBERT")
tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/")

def predict_window(seq, seq_cutoff=39, threshold=0.5):
    global model
    global tokenizer

    predictions = []
    seq_dicts = []

    if len(seq) > seq_cutoff:
        splits = len(seq) - seq_cutoff
        start = time.time()
        logging.info("Splitting sequence into %d windows of length %d" % (splits, seq_cutoff+1))
        for i in range(splits):
            subseq = seq[i:seq_cutoff+i+1]
            inputs = tokenizer(" ".join(subseq), return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.softmax(logits, axis=-1).detach().numpy()

            seq_dict = {
                "startIndex": i,
                "endIndex": seq_cutoff+i,
                "prediction": str(prediction[0][1])
            }
            seq_dicts.append(seq_dict)

        end = time.time()
        logging.info("Prediction took %f seconds" % (end-start))

    return seq_dicts
