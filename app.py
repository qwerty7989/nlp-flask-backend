from flask import Flask, render_template, request
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import re
import pandas as pd
import numpy as np
from time import time


app = Flask(__name__)

with app.app_context():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained('bert-base-uncased')
        model_classification = tf.keras.models.load_model('model/bert_model.h5',custom_objects={'TFBertModel':model})

@app.route("/")
def index():
        return render_template('index.html')

@app.route("/result")
def result():
        start = time()
        input_text = request.args.get('text')

        input_text = re.sub(r'\n+', ' ', input_text)
        id_text = []
        mask_text = []

        bert_inp = tokenizer.encode_plus(input_text, add_special_tokens = True, max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
        id_text.append(bert_inp['input_ids'])
        mask_text.append(bert_inp['attention_mask'])

        id_text = np.array(id_text)
        mask_text = np.array(mask_text)

        result = model_classification.predict([id_text, mask_text])

        return f'You enter {input_text} and it is probably {result} and spent time {time()-start:.3f}'