from tensorflow.keras.models import load_model

model_classification = load_model('models/bert_model.h5')

x_test = test_df['input_ids']
test_mask = test_df['attention_mask']

x_test_array = np.array(x_test.tolist())
test_mask_array = np.array(test_mask.tolist())

y_test = model_classification.predict( [x_test_array, test_mask_array] )

print(y_test)
