
## Purpose of this project

Global Navigation Satellite Systems (GNSS) have become a fundamental technological pillar of our everyday lives and critical infrastructures. From the navigation of autonomous vehicles, through positioning procedures in air traffic, to the time synchronization of financial transactions, numerous applications rely on the precise position and time information provided by GNSS. Parallel to the widespread adoption of these systems, however, intentional interference activities aimed at disrupting or falsifying satellite signals have become increasingly common. Signal spoofing represents a particularly serious security risk, as the counterfeit signals can transmit navigation information that is almost indistinguishable from genuine signals and appears authentic, making the manipulation extremely difficult for the system to detect.

The aim of this project is to research the possiblities of detecting spoofing with machine learning algorithms, especially `XGBoost` algorithm.

## Dependencies 

You need the following modules to be installed and imported
- `numpy`
- `pandas`
- `georinex`
- `sklearn`
- `xgboost`
- `imblearn`
- `optuna`
- `joblib`

The easiest way is to use the `pip install` method. 