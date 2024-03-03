import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import pandas as pd
import numpy as np

class nlp:
    def convert_by_tokenizer(word):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained('bert-base-uncased')
        bert_inp = tokenizer.encode_plus(word, add_special_tokens = True, max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
        ids.append(bert_inp['input_ids'])
        masks.append(bert_inp['attention_mask'])
        return bert_inp['input_ids'], bert_inp['attention_mask']

    def receive_input(input_word):
        #input_word = "Hello, World"
        input_word = [input_word]
        processed_word = pd.DataFrame(data=input_word, columns=['text'])
        ids = []
        masks = []

        processed_word[['input_ids', 'attention_mask']] = processed_word['text'].apply(convert_by_tokenizer).apply(pd.Series)

        processed_word['input_ids'] = np.asarray(processed_word['input_ids'])
        processed_word['attention_mask'] = np.asarray(processed_word['attention_mask'])
        return processed_word

    def calculate_value(input_word):
        processed_word = receive_input(input_word)

        x_test = processed_word['input_ids']
        test_mask = processed_word['attention_mask']
        x_test_array = np.array(x_test.tolist())
        test_mask_array = np.array(test_mask.tolist())

        model_classification = tf.keras.models.load_model('model/bert_model.h5',custom_objects={'TFBertModel':TFBertModel})
        result = model_classification.predict([x_test_array, test_mask_array])
        return result