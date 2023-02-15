import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import time
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    MaxAbsScaler, 
    # RobustScalar,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
    OneHotEncoder, 
    OrdinalEncoder,
)

from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators

from sklearn.base import (
    RegressorMixin, 
    ClassifierMixin,
    TransformerMixin
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    auc,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
    classification_report
)

import warnings
import xgboost
import catboost
import lightgbm

from imblearn.over_sampling import (
    RandomOverSampler, 
    SMOTE, 
    ADASYN, 
    BorderlineSMOTE, 
    SMOTENC, 
    KMeansSMOTE, 
    SVMSMOTE
)

from imblearn.under_sampling import (
    RandomUnderSampler, 
    NearMiss, 
    TomekLinks, 
    EditedNearestNeighbours, 
    RepeatedEditedNearestNeighbours, 
    AllKNN, 
    CondensedNearestNeighbour, 
    OneSidedSelection,
    NeighbourhoodCleaningRule,
    InstanceHardnessThreshold
)

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

TRANSFOMER_METHODS = [
    ("StandardScaler", StandardScaler), 
    ("MinMaxScaler", MinMaxScaler), 
    ("MaxAbsScaler", MaxAbsScaler), 
    # ("RobustScalar", RobustScalar),
    ("Normalizer", Normalizer),
    ("QuantileTransformer", QuantileTransformer),
    ("PowerTransformer", PowerTransformer),
]

OVER_SAMPLING_METHODS = [
    ("RandomOverSampler", RandomOverSampler), 
    ("SMOTE", SMOTE), 
    ("ADASYN", ADASYN), 
    ("BorderlineSMOTE", BorderlineSMOTE), 
    ("SMOTENC", SMOTENC), 
    ("KMeansSMOTE", KMeansSMOTE), 
    ("SVMSMOTE", SVMSMOTE),
]

UNDER_SAMPLING_METHODS = [
    ("RandomUnderSampler", RandomUnderSampler), 
    ("NearMiss", NearMiss), 
    ("TomekLinks", TomekLinks), 
    ("EditedNearestNeighbours", EditedNearestNeighbours), 
    ("RepeatedEditedNearestNeighbours", RepeatedEditedNearestNeighbours),
    ("AllKNN", AllKNN),
    ("CondensedNearestNeighbour", CondensedNearestNeighbour),
    ("OneSidedSelection", OneSidedSelection),
    ("NeighbourhoodCleaningRule", NeighbourhoodCleaningRule),
    ("InstanceHardnessThreshold", InstanceHardnessThreshold),
]

removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    # "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    # "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
    "CategoricalNB",
    "StackingClassifier",
    "NuSVC",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]


REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
REGRESSORS.append(('CatBoostRegressor', catboost.CatBoostRegressor))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
CLASSIFIERS.append(('CatBoostClassifier', catboost.CatBoostClassifier))

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean"))]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)

def get_card_split(df, cols, n=11):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high

class LazierClassifier:
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    classifiers : list, optional (default="all")
        When function is provided, trains the chosen classifier(s).
    """
    def __init__(
        self,
        verbose=0,
        ignore_warnings = True,
        custom_metric = None,
        predictions = False,
        random_state = 42,
        classifiers = "all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers
    
    def fit(self, 
        X, 
        y, 
        test_size = None,
        train_size = None,
        random_state = None,
        shuffle = True, 
        stratify = None,
        sampling_method = None,
        transformer_method = None,
    ):
        """Fit Classification algorithms on data.
        Parameters
        ----------
        X : array-like,
            Features
        y : array-like,
            Class Labels
        test_size : float or int, default = None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
        train_size : float or int, default = None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
        random_state : int, RandomState instance or None, default = None
            Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
        shuffle : bool, default = True
            Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
        stratify : array-like, default = None
            If not None, data is split in a stratified fashion, using this as the class labels.
        sampling_method : str ("over_sample" or "under_sample"), default = None
            Used to define if you want to over_sample or under_sample the dataset.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """

        sampling_method_list = []
        if sampling_method == None:
            sampling_method_list = []
        elif sampling_method == "over_sample":
            sampling_method_list = OVER_SAMPLING_METHODS
        elif sampling_method == "under_sample":
            sampling_method_list = UNDER_SAMPLING_METHODS
        else:
            sampling_method_list = sampling_method
        
        transformer_method_list = []
        if transformer_method == None:
            transformer_method_list = []
        elif transformer_method == "all":
            transformer_method_list = TRANSFOMER_METHODS
        else:
            transformer_method_list = transformer_method

        stratify_method = None
        if stratify == None:
            stratify_method = None
        else:
            stratify_method = y
        
        le = preprocessing.LabelEncoder()
        le = le.fit(y)
        y = le.transform(y)
        print(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, train_size = train_size, stratify = stratify_method, shuffle = shuffle, random_state = random_state)
        
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS
        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")
        
        if sampling_method == None and transformer_method == None:
            for name, model in tqdm(self.classifiers):
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("classifier", model(random_state = self.random_state)),
                            ]
                        )
                    else:
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("classifier", model()),
                            ]
                        )

                    pipe.fit(X_train, y_train)
                    self.models[name] = pipe
                    y_pred = pipe.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred, normalize=True)
                    b_accuracy = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred)
                    except Exception as exception:
                        roc_auc = None
                        if self.ignore_warnings is False:
                            print("ROC AUC couldn't be calculated for " + name)
                            print(exception)
                    names.append(name)
                    Accuracy.append(accuracy)
                    B_Accuracy.append(b_accuracy)
                    ROC_AUC.append(roc_auc)
                    F1.append(f1)
                    TIME.append(time.time() - start)
                    if self.custom_metric is not None:
                        custom_metric = self.custom_metric(y_test, y_pred)
                        CUSTOM_METRIC.append(custom_metric)
                    if self.verbose > 0:
                        if self.custom_metric is not None:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    self.custom_metric.__name__: custom_metric,
                                    "Time taken": time.time() - start,
                                }
                            )
                        else:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    "Time taken": time.time() - start,
                                }
                            )
                    if self.predictions:
                        predictions[name] = y_pred
                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)
            if self.custom_metric is None:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        "Time Taken": TIME,
                    }
                )
            else:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        self.custom_metric.__name__: CUSTOM_METRIC,
                        "Time Taken": TIME,
                    }
                )
            scores = scores.sort_values(by = "Balanced Accuracy", ascending = False).set_index("Model")

            if self.predictions:
                predictions_df = pd.DataFrame.from_dict(predictions)

            return scores, predictions_df if self.predictions is True else scores
        elif sampling_method != None and transformer_method == None:
            try:
                for sampling_method_name, sampling_method_model in tqdm(sampling_method_list):
                    print(sampling_method_name)
                    for name, model in tqdm(self.classifiers):
                        start = time.time()
                        if "random_state" in model().get_params().keys():
                            pipe = Pipeline(
                                steps=[
                                    ("preprocessor", preprocessor),
                                    ("sampling_method", sampling_method_model()), 
                                    ("classifier", model(random_state = self.random_state)),
                                ]
                            )
                        else:
                            pipe = Pipeline(
                                steps=[
                                    ("preprocessor", preprocessor),
                                    ("sampling_method", sampling_method_model()), 
                                    ("classifier", model()),
                                ]
                            )
                        pipe.fit(X_train, y_train)
                        self.models[name + " (" + sampling_method_name + ")"] = pipe
                        # print(self.models)
                        y_pred = pipe.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred, normalize=True)
                        b_accuracy = balanced_accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted")
                        try:
                            roc_auc = roc_auc_score(y_test, y_pred)
                        except Exception as exception:
                            roc_auc = None
                            if self.ignore_warnings is False:
                                print("ROC AUC couldn't be calculated for " + name)
                                print(exception)
                        names.append(name + " (" + sampling_method_name + ")")
                        Accuracy.append(accuracy)
                        B_Accuracy.append(b_accuracy)
                        ROC_AUC.append(roc_auc)
                        F1.append(f1)
                        TIME.append(time.time() - start)
                        if self.custom_metric is not None:
                            custom_metric = self.custom_metric(y_test, y_pred)
                            CUSTOM_METRIC.append(custom_metric)
                        if self.verbose > 0:
                            if self.custom_metric is not None:
                                print(
                                    {
                                        "Model": name + " (" + sampling_method_name + ")",
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        self.custom_metric.__name__: custom_metric,
                                        "Time taken": time.time() - start,
                                    }
                                )
                            else:
                                print(
                                    {
                                        "Model": name + " (" + sampling_method_name + ")",
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        "Time taken": time.time() - start,
                                    }
                                )
                        if self.predictions:
                            predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " (" + sampling_method_name + ")" + " model failed to execute")
                    print(exception)
            if self.custom_metric is None:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        "Time Taken": TIME,
                    }
                )
            else:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        self.custom_metric.__name__: CUSTOM_METRIC,
                        "Time Taken": TIME,
                    }
                )
            scores = scores.sort_values(by = "Balanced Accuracy", ascending = False).set_index("Model")

            if self.predictions:
                predictions_df = pd.DataFrame.from_dict(predictions)

            return scores, predictions_df if self.predictions is True else scores
        elif sampling_method == None and transformer_method != None:
            try:
                for transformer_method_name, transformer_method_model in tqdm(transformer_method_list):
                    print(transformer_method_name)
                    for name, model in tqdm(self.classifiers):
                        start = time.time()
                        if "random_state" in model().get_params().keys():
                            pipe = Pipeline(
                                steps=[
                                    ("preprocessor", preprocessor),
                                    ("transformer", transformer_method_model()), 
                                    ("classifier", model(random_state = self.random_state)),
                                ]
                            )
                        else:
                            pipe = Pipeline(
                                steps=[
                                    ("preprocessor", preprocessor),
                                    ("transformer", transformer_method_model()), 
                                    ("classifier", model()),
                                ]
                            )
                        pipe.fit(X_train, y_train)
                        self.models[name + " (" + transformer_method_name + ")"] = pipe
                        # print(self.models)
                        y_pred = pipe.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred, normalize=True)
                        b_accuracy = balanced_accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted")
                        try:
                            roc_auc = roc_auc_score(y_test, y_pred)
                        except Exception as exception:
                            roc_auc = None
                            if self.ignore_warnings is False:
                                print("ROC AUC couldn't be calculated for " + name)
                                print(exception)
                        names.append(name + " (" + transformer_method_name + ")")
                        Accuracy.append(accuracy)
                        B_Accuracy.append(b_accuracy)
                        ROC_AUC.append(roc_auc)
                        F1.append(f1)
                        TIME.append(time.time() - start)
                        if self.custom_metric is not None:
                            custom_metric = self.custom_metric(y_test, y_pred)
                            CUSTOM_METRIC.append(custom_metric)
                        if self.verbose > 0:
                            if self.custom_metric is not None:
                                print(
                                    {
                                        "Model": name + " (" + transformer_method_name + ")",
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        self.custom_metric.__name__: custom_metric,
                                        "Time taken": time.time() - start,
                                    }
                                )
                            else:
                                print(
                                    {
                                        "Model": name + " (" + transformer_method_name + ")",
                                        "Accuracy": accuracy,
                                        "Balanced Accuracy": b_accuracy,
                                        "ROC AUC": roc_auc,
                                        "F1 Score": f1,
                                        "Time taken": time.time() - start,
                                    }
                                )
                        if self.predictions:
                            predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " (" + transformer_method_name + ")" + " model failed to execute")
                    print(exception)
            if self.custom_metric is None:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        "Time Taken": TIME,
                    }
                )
            else:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        self.custom_metric.__name__: CUSTOM_METRIC,
                        "Time Taken": TIME,
                    }
                )
            scores = scores.sort_values(by = "Balanced Accuracy", ascending = False).set_index("Model")

            if self.predictions:
                predictions_df = pd.DataFrame.from_dict(predictions)

            return scores, predictions_df if self.predictions is True else scores
        elif sampling_method != None and transformer_method != None:
            try:
                for transformer_method_name, transformer_method_model in tqdm(transformer_method_list):
                    for sampling_method_name, sampling_method_model in tqdm(sampling_method_list):
                        print(transformer_method_name, sampling_method_name)
                        for name, model in tqdm(self.classifiers):
                            start = time.time()
                            if "random_state" in model().get_params().keys():
                                pipe = Pipeline(
                                    steps=[
                                        ("preprocessor", preprocessor),
                                        ("sampling_method", sampling_method_model()), 
                                        ("transformer", transformer_method_model()), 
                                        ("classifier", model(random_state = self.random_state)),
                                    ]
                                )
                            else:
                                pipe = Pipeline(
                                    steps=[
                                        ("preprocessor", preprocessor),
                                        ("sampling_method", sampling_method_model()), 
                                        ("transformer", transformer_method_model()), 
                                        ("classifier", model()),
                                    ]
                                )
                            pipe.fit(X_train, y_train)
                            self.models[name + " (" + transformer_method_name + ")"] = pipe
                            # print(self.models)
                            y_pred = pipe.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred, normalize=True)
                            b_accuracy = balanced_accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average="weighted")
                            try:
                                roc_auc = roc_auc_score(y_test, y_pred)
                            except Exception as exception:
                                roc_auc = None
                                if self.ignore_warnings is False:
                                    print("ROC AUC couldn't be calculated for " + name)
                                    print(exception)
                            names.append(name + " (" + sampling_method_name + ") (" + transformer_method_name + ")")
                            Accuracy.append(accuracy)
                            B_Accuracy.append(b_accuracy)
                            ROC_AUC.append(roc_auc)
                            F1.append(f1)
                            TIME.append(time.time() - start)
                            if self.custom_metric is not None:
                                custom_metric = self.custom_metric(y_test, y_pred)
                                CUSTOM_METRIC.append(custom_metric)
                            if self.verbose > 0:
                                if self.custom_metric is not None:
                                    print(
                                        {
                                            "Model": name + " (" + sampling_method_name + ") (" + transformer_method_name + ")",
                                            "Accuracy": accuracy,
                                            "Balanced Accuracy": b_accuracy,
                                            "ROC AUC": roc_auc,
                                            "F1 Score": f1,
                                            self.custom_metric.__name__: custom_metric,
                                            "Time taken": time.time() - start,
                                        }
                                    )
                                else:
                                    print(
                                        {
                                            "Model": name + " (" + sampling_method_name + ") (" + transformer_method_name + ")",
                                            "Accuracy": accuracy,
                                            "Balanced Accuracy": b_accuracy,
                                            "ROC AUC": roc_auc,
                                            "F1 Score": f1,
                                            "Time taken": time.time() - start,
                                        }
                                    )
                            if self.predictions:
                                predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " (" + sampling_method_name + ") (" + transformer_method_name + ")" + " model failed to execute")
                    print(exception)
            if self.custom_metric is None:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        "Time Taken": TIME,
                    }
                )
            else:
                scores = pd.DataFrame(
                    {
                        "Model": names,
                        "Accuracy": Accuracy,
                        "Balanced Accuracy": B_Accuracy,
                        "ROC AUC": ROC_AUC,
                        "F1 Score": F1,
                        self.custom_metric.__name__: CUSTOM_METRIC,
                        "Time Taken": TIME,
                    }
                )
            scores = scores.sort_values(by = "Balanced Accuracy", ascending = False).set_index("Model")

            if self.predictions:
                predictions_df = pd.DataFrame.from_dict(predictions)

            return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models