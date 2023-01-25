
from predictor.lazier_predictor import LazierClassifier

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target
    clf = LazierClassifier(verbose = 0, ignore_warnings = False, custom_metric = None)
    
    # simple lazypredict
    models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True)
    models.to_csv("lazier_models_cancer.csv")
    predictions.to_csv("lazier_predictions_cancer.csv")
    
    # lazypredict + oversampling
    models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "over_sample")
    models.to_csv("lazier_models_over_sample_cancer.csv")
    predictions.to_csv("lazier_predictions_over_sample_cancer.csv")
    
    # lazypredict + undersampling
    models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "under_sample")
    models.to_csv("lazier_models_under_sample_cancer.csv")
    predictions.to_csv("lazier_predictions_under_sample_cancer.csv")

    # lazypredict + transform
    models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, transformer_method = "all")
    models.to_csv("lazier_models_transformer_cancer.csv")
    predictions.to_csv("lazier_predictions_transformer_cancer.csv")

    # lazypredict + transform + oversampling
    models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "over_sample", transformer_method = "all")
    models.to_csv("lazier_models_over_sample_transformer_cancer.csv")
    predictions.to_csv("lazier_predictions_over_sample_transformer_cancer.csv")

    # lazypredict + transform + undersampling
    models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True, sampling_method = "under_sample", transformer_method = "all")
    models.to_csv("lazier_models_under_sample_transformer_cancer.csv")
    predictions.to_csv("lazier_predictions_under_sample_transformer_cancer.csv")

    # testing for multi-class
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data.data
    y = data.target
    clf = LazierClassifier(verbose = 0, ignore_warnings = False, custom_metric = None)
    
    models, predictions = clf.fit(X, y, test_size = 0.20, random_state = None, shuffle = True, stratify = True)
    models.to_csv("lazier_models_wine.csv")
    predictions.to_csv("lazier_predictions_wine.csv")