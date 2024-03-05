from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from time import time

app = Flask(__name__)

with app.app_context():
        BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        BERT_model = TFBertModel.from_pretrained('bert-base-uncased')
        BERT_classification = tf.keras.models.load_model('model/bert_model.h5', custom_objects={'TFBertModel': BERT_model})

@app.route("/")
def index():
        return render_template('index.html')

@app.route("/result")
def result():
        start_time = time()

        # Recieve & Clean data
        input_text = request.args.get('text')
        input_text = re.sub(r'\n+', ' ', input_text)

        # Encoding data
        id_text = []
        mask_text = []
        BERT_INP = BERT_tokenizer.encode_plus(input_text, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, truncation=True)

        id_text.append(BERT_INP['input_ids'])
        mask_text.append(BERT_INP['attention_mask'])

        id_text = np.array(id_text)
        mask_text = np.array(mask_text)

        # Predict
        result = BERT_classification.predict([id_text, mask_text])

        return f'You enter {input_text} and it is probably {result} and spent time {time()-start_time:.3f}'