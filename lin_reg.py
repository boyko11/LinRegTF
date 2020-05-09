import tensorflow as tf
from service.data_service import DataService
import numpy as np

data = DataService.load_csv("data/blood_pressure_cengage.csv")

def get_data():
    feature_data = data[:, :-1]
    # add 1 at the beginning for the bias
    feature_data = np.insert(feature_data, 0, 1, axis=1)
    labels = data[:, -1]
    return {"Age": feature_data[:, 0], "Weight": feature_data[:, 1]}, labels


def get_feature_data():
    feature_data, _ = get_data()
    return feature_data


def get_labels():
    _, labels = get_data()
    return labels


age_column = tf.feature_column.numeric_column('Age')
weight_column = tf.feature_column.numeric_column('Weight')

lin_reg = tf.estimator.LinearRegressor(feature_columns=[age_column, weight_column])

train_steps = 20000
lin_reg.train(input_fn=get_data, steps=train_steps)

test_steps = 1
metrics = lin_reg.evaluate(input_fn=get_data, steps=test_steps)
print(metrics)

predictions = lin_reg.predict(input_fn=get_feature_data)

print("Label, Prediction")
train_labels = get_labels()
for record_index, prediction in enumerate(predictions):
    label = train_labels[record_index]
    predicted_label = np.rint(prediction['predictions'][0])
    print(label, predicted_label)
    if record_index == 10:
        break

print("Done.")