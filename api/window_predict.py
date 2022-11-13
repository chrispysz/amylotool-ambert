from transformers import BertForSequenceClassification, BertTokenizer
import torch
import logging
import time


def predict_window(seq):

    start = time.process_time()

    seq_cutoff = 39

    threshold = 0.5

    model = BertForSequenceClassification.from_pretrained(
        "./models/AmBERT")
    tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/")

    current_label = 0

    predictions = []
    sequences = []
    spec_preds = []
    seq_dicts = []

    seq_proper = ''
    for aa in seq:
        seq_proper += aa+' '

    if len(seq) > seq_cutoff:
        splits = len(seq)-seq_cutoff
        for i in range(splits):
            subseq = seq[i:seq_cutoff+i+1]
            inputs = tokenizer(" ".join(subseq), return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.softmax(logits, axis=-1).detach().numpy()
            if prediction[0][1] > threshold:
                predictions.append('1')
            else:

                predictions.append('0')

            seq_dict = {
                "sequence": subseq,
                "prediction": str(prediction[0][1])
            }
            seq_dicts.append(seq_dict)

            logging.warn("Processed " + str(i+1)+"/"+str(splits))

    elapsed = time.process_time() - start
    logging.warn(msg="Prediction processed in "+str(elapsed)+" seconds")
    if '1' in predictions:
        return ("Positive", seq_dicts)

    else:
        return ("Negative", seq_dicts)
