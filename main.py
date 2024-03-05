import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import re
import pandas as pd
import numpy as np

def encode_text(text):
    bert_inp = tokenizer.encode_plus(text, add_special_tokens = True, max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
    id_text.append(bert_inp['input_ids'])
    mask_text.append(bert_inp['attention_mask'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

input_text = "Hello \n\n\n Hi"
input_text = re.sub(r'\n+', ' ', input_text)
id_text = []
mask_text = []

encode_text(input_text)

id_text = np.array(id_text)
mask_text = np.array(mask_text)

model_classification = tf.keras.models.load_model('model/bert_model.h5',custom_objects={'TFBertModel':TFBertModel})

result = model_classification.predict([id_text, mask_text])

print(result)