import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import pandas as pd
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def testid(word):
    bert_inp = tokenizer.encode_plus(word, add_special_tokens = True, max_length = 512, padding = 'max_length', return_attention_mask = True, truncation = True)
    test_ids.append(bert_inp['input_ids'])
    test_masks.append(bert_inp['attention_mask'])
    return bert_inp['input_ids'], bert_inp['attention_mask']

input_word = "Hello, World"
input_word = [input_word]
test_df = pd.DataFrame(data=input_word, columns=['text'])
test_ids = []
test_masks = []

test_df[['input_ids', 'attention_mask']] = test_df['text'].apply(testid).apply(pd.Series)

test_df['input_ids'] = np.asarray(test_df['input_ids'])
test_df['attention_mask'] = np.asarray(test_df['attention_mask'])

x_test = test_df['input_ids']
test_mask = test_df['attention_mask']

x_test_array = np.array(x_test.tolist())
test_mask_array = np.array(test_mask.tolist())
print(x_test_array)

model_classification = tf.keras.models.load_model('model/bert_model.h5',custom_objects={'TFBertModel':TFBertModel})

result = model_classification.predict([x_test_array, test_mask_array])

print(result)