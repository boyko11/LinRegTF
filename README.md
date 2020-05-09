Linear Regresssion with Tensorflow

## Dataset
The simple cengage systolic blood pressure dataset:  
https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html

## Purpose
Since I implemented linear regression from scratch,  
this is to see if Tensorflow can make it easier

#### Run verified with python 3.7 and numpy 1.18.1
python lin_reg.py

## Loss Per Tensorflow Step (Learning Curve)
![Loss per Tensorflow Step](plots/learning_curve.png)

## Predicted vs Actual
![Predicted vs Actual](plots/predicted_vs_actual.png)

## Prediction Results over the Training set

|Age|Weight|Actual Blood Pressure|Predicted Blood Pressure|Loss
|:-------:|:---:|:--------------------:|:---------------------:|:------:|
|52.0|173.0|132.0|128.0|4.0|
|59.0|184.0|143.0|142.0|1.0|
|67.0|194.0|153.0|159.0|6.0|
|73.0|211.0|162.0|171.0|9.0|
|64.0|196.0|154.0|153.0|1.0|
|74.0|220.0|168.0|173.0|5.0|
|54.0|188.0|137.0|132.0|5.0|
|61.0|188.0|149.0|146.0|3.0|
|65.0|207.0|159.0|155.0|4.0|
|46.0|167.0|128.0|116.0|12.0|
|72.0|217.0|166.0|169.0|3.0|