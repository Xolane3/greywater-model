AMANZI IMPILO — JUPYTER/PANDAS ML-READY DATASET
====================================================

This package contains synthetic data for a Proof of Concept.

IMPORTANT:
All data is dummy/synthetic training data. It is NOT measured Amanzi
Impilo field data and should not be presented as actual system performance.

MODELS
------

1. POTABLE WATER SAVINGS REGRESSION
Goal:
Predict how many litres of potable water the system can save.

Target:
potable_saved_l

Feature file:
02_water_savings_features.csv

Train/test:
04_water_savings_train_X.csv
05_water_savings_train_y.csv
06_water_savings_test_X.csv
07_water_savings_test_y.csv

Leakage avoided:
The direct outcome columns potable_before_l, potable_after_l and
potable_saved_l are not included in the feature file.

2. COUNTERFACTUAL POTABLE WATER MODEL
Goal:
Predict how much potable water would be required if the greywater
system did not exist.

Target:
potable_before_l

Feature file:
08_counterfactual_features.csv

Train/test:
10_counterfactual_train_X.csv
11_counterfactual_train_y.csv
12_counterfactual_test_X.csv
13_counterfactual_test_y.csv

3. SENSOR REPLACEMENT CLASSIFIER
Goal:
Predict whether a sensor needs replacement/maintenance.

Target:
replacement_required
0 = no replacement warning
1 = replacement/maintenance warning

Feature file:
15_sensor_replacement_features.csv

Train/test:
17_sensor_replacement_train_X.csv
18_sensor_replacement_train_y.csv
19_sensor_replacement_test_X.csv
20_sensor_replacement_test_y.csv

4. SENSOR REMAINING USEFUL LIFE
Goal:
Predict how many days of useful operation a sensor may have remaining.

Target:
remaining_useful_life_days

Train/test:
17_sensor_replacement_train_X.csv
22_sensor_remaining_life_train_y.csv
19_sensor_replacement_test_X.csv
23_sensor_remaining_life_test_y.csv

DATA SPLIT
----------
The split is chronological: approximately 80% of the earliest dates are
training data and the latest 20% are testing data. This is preferable to
randomly mixing future observations into training data for a time-dependent
water/sensor project.

FILES
-----
01-07: Water savings model
08-13: Counterfactual potable-water model
14-20: Sensor replacement model
21-23: Sensor remaining-life model
24: Combined daily dataset
25: Python/Jupyter starter code

NEXT STEP FOR THE REAL POC
--------------------------
When Amanzi Impilo starts producing real sensor readings, replace the
synthetic CSVs with actual observations. For sensor-life prediction, keep
maintenance dates, calibration dates, fault events and actual replacement
dates. That will make the replacement model evidence-based instead of
synthetic.
