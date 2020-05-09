import tensorflow as tf
from service.data_service import DataService
import numpy as np
from service.plot_service import PlotService

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

steps_history = []
loss_history = []

start_steps = 1
max_steps = 10000
current_steps = start_steps
step_increment = 50

while current_steps < max_steps:
    print(current_steps)
    lin_reg = tf.estimator.LinearRegressor(feature_columns=[age_column, weight_column])
    lin_reg.train(input_fn=get_data, steps=current_steps)
    metrics = lin_reg.evaluate(input_fn=get_data, steps=current_steps)
    steps_history.append(current_steps)
    loss_history.append(metrics['loss'])
    current_steps += step_increment

print(metrics)

predictions = lin_reg.predict(input_fn=get_feature_data)

print("|Age|Weight|Actual Blood Pressure|Predicted Blood Pressure|Loss|")
print("|:-------:|:---:|:--------------------:|:---------------------:|")

train_records = data[:, :-1]
train_labels = get_labels()
predicted_labels = []
for record_index, prediction in enumerate(predictions):
    age = train_records[record_index, 0]
    weight = train_records[record_index, 1]
    label = train_labels[record_index]
    predicted_label = np.rint(prediction['predictions'][0])
    predicted_labels.append(predicted_label)
    loss = np.abs(label - predicted_label)

    print("|{0}|{1}|{2}|{3}|{4}|".format(age, weight, label, predicted_label, loss))
    if record_index == 10:
        break

PlotService.plot_line(
    x=steps_history,
    y=loss_history,
    x_label="Tensorflow Steps",
    y_label="Loss",
    title="Loss for Steps", ylim=(0, 250))

PlotService.plot3d_scatter_compare(train_records, train_labels, predicted_labels, labels=['Age', 'Weight', 'BP'],
                                   title="Actual vs Projected")

print("Done.")