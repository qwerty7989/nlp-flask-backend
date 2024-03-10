from flask import Flask, render_template, request
from transformers import AutoTokenizer, TFAutoModel
import sentencepiece
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from time import time

app = Flask(__name__)

with app.app_context():
        # Load Model
        BERT_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        BERT_model = TFAutoModel.from_pretrained('bert-base-uncased')
        BERT_classification = tf.keras.models.load_model('model/bert_model_auto.h5', custom_objects={'TFBertModel': BERT_model})

        DEBERTA_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small', use_fast=False)
        DEBERTA_model = TFAutoModel.from_pretrained('microsoft/deberta-v3-small')
        DEBERTA_classification = tf.keras.models.load_model('model/deberta_model_7.h5', custom_objects={'TFDebertaV2Model': DEBERTA_model})

        ROBERTA_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        ROBERTA_model = TFAutoModel.from_pretrained('FacebookAI/roberta-base')
        ROBERTA_classification = tf.keras.models.load_model('model/roberta_model_2.h5', custom_objects={'TFRobertaModel': ROBERTA_model})

@app.route("/")
def index():
        return render_template('index.html')

@app.route("/result")
def result():
        start_time = time()

        # Recieve & Clean data
        input_text = request.args.get('text')
        input_text = re.sub(r'\n+', ' ', input_text)

        # Recieve chosen model
        chosen_model = request.args.get('model')

        # Encoding data
        id_text = []
        mask_text = []

        if chosen_model == "bert":
                MODEL_INP = BERT_tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, truncation=True)
        elif chosen_model == "deberta":
                MODEL_INP = DEBERTA_tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, truncation=True)
        elif chosen_model == "roberta":
                MODEL_INP = ROBERTA_tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, truncation=True)

        id_text.append(MODEL_INP['input_ids'])
        mask_text.append(MODEL_INP['attention_mask'])

        id_text = np.array(id_text)
        mask_text = np.array(mask_text)

        # Predict
        if chosen_model == "bert":
                result = BERT_classification.predict([id_text, mask_text])
        elif chosen_model == "deberta":
                result = DEBERTA_classification.predict([id_text, mask_text])
        elif chosen_model == "roberta":
                result = ROBERTA_classification.predict([id_text, mask_text])

        return f'You enter {input_text} and it is probably {result} and spent time {time()-start_time:.3f}'