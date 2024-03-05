import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import re
import pandas as pd
import numpy as np

class nlp:
    def __init__(self):
        self.id_text = []
        self.mask_text = []

    def encoding_input(self, input_text):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained('bert-base-uncased')
        bert_inp = tokenizer.encode_plus(input_text, add_special_tokens = True, max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
        self.id_text.append(bert_inp['input_ids'])
        self.mask_text.append(bert_inp['attention_mask'])

    def process_input(self, input_text):
        input_text = re.sub(r'\n+', ' ', input_text)

        self.encoding_input(input_text)

        self.id_text = np.array(self.id_text)
        self.mask_text = np.array(self.mask_text)

    def calculate_value(self, input_text):
        self.process_input(input_text)

        model_classification = tf.keras.models.load_model('model/bert_model.h5',custom_objects={'TFBertModel':TFBertModel})
        result = model_classification.predict([self.id_text, self.mask_text])
        return result
