# Lazier Predict

Basically lazypredict but lazier.

Additional Feature/Why it exists even when lazypredict already does:
1. Try combinations of transformations (StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder) on dataset. 
2. Try combinations of oversampling (RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SMOTENC, KMeansSMOTE, SVMSMOTE) and undersampling (RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold).
3. Try combinations of both transformers and sampling.
4. Add back models like: Catboost, GradientBoostingClassifier, MLPClassifier

Other than that, I guess it's also for people that don't care too much about the time it takes to find the best model and they are lazier.

Usage for classifier:
```python
from predictor.lazier_predictor import LazierClassifier
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
clf = LazierClassifier(verbose = 0, ignore_warnings = False, custom_metric = None)

# simple lazierpredict classifier
models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True)
models.to_csv("lazier_models_cancer.csv")
predictions.to_csv("lazier_predictions_cancer.csv")

# lazierpredict classifier + oversampling
models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "over_sample")
models.to_csv("lazier_models_over_sample_cancer.csv")
predictions.to_csv("lazier_predictions_over_sample_cancer.csv")

# lazierpredict classifier + undersampling
models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "under_sample")
models.to_csv("lazier_models_under_sample_cancer.csv")
predictions.to_csv("lazier_predictions_under_sample_cancer.csv")

# lazierpredict classifier + transform
models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, transformer_method = "all")
models.to_csv("lazier_models_transformer_cancer.csv")
predictions.to_csv("lazier_predictions_transformer_cancer.csv")

# lazierpredict classifier + transform + oversampling
models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "over_sample", transformer_method = "all")
models.to_csv("lazier_models_over_sample_transformer_cancer.csv")
predictions.to_csv("lazier_predictions_over_sample_transformer_cancer.csv")

# lazierpredict classifier + transform + undersampling
models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "under_sample", transformer_method = "all")
models.to_csv("lazier_models_under_sample_transformer_cancer.csv")
predictions.to_csv("lazier_predictions_under_sample_transformer_cancer.csv")
```

Usage for regressor:
```python
from predictor.lazier_predictor import LazierRegressor
from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target
print(X.shape, y.shape)
reg = LazierRegressor(verbose = 0,ignore_warnings = False, custom_metric = None)
models, predictions = reg.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, transformer_method = "all")
models.to_csv("lazier_models_boston.csv")
predictions.to_csv("lazier_predictions_boston.csv")
```

Working on:
1. Adding more metrics
3. Hyperparameter Tuning (best models get tuned)
4. Pipeline Stacking (best models get stacked)
5. etc.

I would advise running this on someone else's PC/HPC.
