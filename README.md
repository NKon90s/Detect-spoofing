
## Purpose of this project

Global Navigation Satellite Systems (GNSS) have become a fundamental technological pillar of our everyday lives and critical infrastructures. From the navigation of autonomous vehicles, through positioning procedures in air traffic, to the time synchronization of financial transactions, numerous applications rely on the precise position and time information provided by GNSS. Parallel to the widespread adoption of these systems, however, intentional interference activities aimed at disrupting or falsifying satellite signals have become increasingly common. Signal spoofing represents a particularly serious security risk, as the counterfeit signals can transmit navigation information that is almost indistinguishable from genuine signals and appears authentic, making the manipulation extremely difficult for the system to detect.

The aim of this project was to research the possiblities of detecting spoofing with machine learning algorithms, especially `XGBoost` algorithm. The results are promising. Machnie learning can be of help in identifying spoofed signals, especially combined with other detection techniques. 

## Dependencies 

The following python modules were used for the project
- `Numpy`
- `Pandas`
- `Georinex`
- `Scikit-learn`
- `Xgboost`
- `Imblearn`
- `Optuna`
- `Joblib`
- `Matplotlib`

The easiest way is to install and use them is the `pip install` method. E.g: `pip install pandas`

## Dataset 

The `rinex_conversion.py` file can convert RINEX files to CSV, what we can feed to our machine learning modell. 

## How to Run

## Results

## Folder Structure