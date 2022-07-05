import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from baseline import compute_class_weight
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score, make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from collections import Counter
import logging

# try 3 linear classifiers with the AQ scores as features
# use grid search for each classifier
# either use combined predictions (average) or use the 3 seed based predictions
#

svm_parameters = {'kernel': ('linear', 'rbf'),
                  'class_weight': ('balanced', *[{1: 2.5}, {1: 3}, {1: 3.5}]),
                  'C': [1, 10, 100],
                  'gamma': [10, 1, 0.1, 0.01, 0.001, 'scale']}

xboost_parameters = {
    "colsample_bytree": [0.3, 0.5, 0.7],
    "gamma": [0.0, 0.5],
    "learning_rate": [0.03, 0.05, 0.1, 0.3],  # default 0.1
    "max_depth": [2, 3, 4, 5, 6],  # default 3
    "n_estimators": [100, 150, 50],  # default 100
    "subsample": [0.6, 0.4]
}
regression_parameters = {
    "C": np.logspace(-3, 3, 7),
    "penalty": ["l1", "l2"],
    "class_weight": ["balanced"]
}


def xg_boost(train_x, test_x, train_y, test_y):
    # feature_cols = list(train_x.num_cols)

    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    weights = compute_class_weight(y=train_y, class_weight="balanced", classes=np.unique(train_y))

    classifier = XGBClassifier(objective="binary:logistic")
    weights_instance = [weights[i] for i in train_y]

    grid_search = GridSearchCV(estimator=classifier, param_grid=xboost_parameters, cv=3,
                               scoring=make_scorer(balanced_accuracy_score), n_jobs=1)
    grid_search_results = grid_search.fit(train_x, train_y, sample_weight=weights_instance)

    logging.info('grid search cv', str(grid_search_results.best_score_))
    for param_name in sorted(xboost_parameters.keys()):
        logging.info("%s: %r" % (param_name, str(grid_search_results.best_params_[param_name])))

    y_pred = grid_search.predict(test_x)
    # plot_shap_values(classifier=classifier, class_names=get_class_names(label), dataset=test_x,
    #                 output="%s_all_classes_split%d" % (label, i))
    # logging.info(classification_report(y_true=test_y, y_pred=y_pred))
    return classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)


def logistic_regression(train_x, test_x, train_y, test_y):
    classifier = LogisticRegression(class_weight="balanced")
    grid_search = GridSearchCV(estimator=classifier, param_grid=regression_parameters, cv=3,
                               scoring=make_scorer(balanced_accuracy_score), n_jobs=1)
    grid_search_results = grid_search.fit(train_x, train_y)

    logging.info('grid search cv', str(grid_search_results.best_score_))
    for param_name in sorted(regression_parameters.keys()):
        logging.info("%s: %r" % (param_name, str(grid_search_results.best_params_[param_name])))
    y_pred = grid_search.predict(test_x)

    # plot_feature_importance(classifier, train_x.columns)
    report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)

    return report


def svm(train_x, test_x, train_y, test_y):
    classifier = SVC()
    grid_search = GridSearchCV(estimator=classifier, param_grid=svm_parameters, cv=3,
                               scoring=make_scorer(balanced_accuracy_score), n_jobs=1)
    grid_search_results = grid_search.fit(train_x, train_y)

    logging.info('grid search cv', str(grid_search_results.best_score_))
    for param_name in sorted(svm_parameters.keys()):
        logging.info("%s: %r" % (param_name, str(grid_search_results.best_params_[param_name])))
    y_pred = grid_search.predict(test_x)
    report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    return report


def transform_qual_score(x):
    if x == "1":
        return 1
    else:
        return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    #data = pd.read_csv(
    #    "/Users/falkne/PycharmProjects/argQuality_adapters/data/paralellInferenceResults/5/regroom.tsv",
    #    sep="\t")
    data = pd.read_csv("/mount/arbeitsdaten/tcl/Users/falk/DQI/data/regroom.tsv", sep="\t")
    print(data.columns)
    dataforgeneral = data.drop("MODFORQUAL", axis=1)

    dataforqual = data[
        ((data["MODFORQUAL"] == '1') & (data["MODINTERVENTION"] == 1)) | (data["MODINTERVENTION"] == 0) | (
                data["MODFORQUAL"] == '0')]
    dataforqual['MODFORQUAL'] = dataforqual['MODFORQUAL'].apply(transform_qual_score)
    dataforqual.drop("MODINTERVENTION", axis=1, inplace=True)
    dataforqual = dataforqual.drop(dataforqual.query('MODFORQUAL == 0').sample(frac=0.6).index)

    # y = data["MODINTERVENTION"]
    # X = data.drop("MODINTERVENTION", axis=1)
    total_scores = {"xgboost": {"f1_macro": [], "f1_positive": []},
                    "logisticRegression": {"f1_macro": [], "f1_positive": []},
                    "svm": {"f1_macro": [], "f1_positive": []}}
    counter = 0
    kf = StratifiedKFold(n_splits=5)
    logging.info("#########.............. general moderation.....################")
    for train_index, test_index in kf.split(dataforgeneral, dataforgeneral["MODINTERVENTION"]):
        train, test = dataforgeneral.iloc[train_index], dataforgeneral.iloc[test_index]
        train_x, test_x = train.drop("MODINTERVENTION", axis=1), test.drop("MODINTERVENTION", axis=1)
        # train_x, test_x = train.drop("MODFORQUAL", axis=1), test.drop("MODFORQUAL", axis=1)

        # scale features
        scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        train_y, test_y = train["MODINTERVENTION"], test["MODINTERVENTION"]
        logging.info("###XGBOOST###")

        report = xg_boost(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)
        total_scores["xgboost"]["f1_macro"].append(report["macro avg"]["f1-score"])
        total_scores["xgboost"]["f1_positive"].append(report["1"]["f1-score"])
        logging.info("F1 macro on split %d: %s" % (counter, report["macro avg"]["f1-score"]))
        logging.info("F1 positive on split %d: %s" % (counter, report["1"]["f1-score"]))

        logging.info("###SVM###")
        report = svm(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)

        total_scores["svm"]["f1_macro"].append(report["macro avg"]["f1-score"])
        total_scores["svm"]["f1_positive"].append(report["1"]["f1-score"])
        logging.info("F1 macro on split %d: %s" % (counter, report["macro avg"]["f1-score"]))
        logging.info("F1 positive on split %d: %s" % (counter, report["1"]["f1-score"]))

        logging.info("###logistic Regression###")
        report = logistic_regression(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)

        total_scores["logisticRegression"]["f1_macro"].append(report["macro avg"]["f1-score"])
        total_scores["logisticRegression"]["f1_positive"].append(report["1"]["f1-score"])

        logging.info("F1 macro on split %d: %s" % (counter, report["macro avg"]["f1-score"]))
        logging.info("F1 positive on split %d: %s" % (counter, report["1"]["f1-score"]))
        counter += 1

    logging.info("----------------")
    logging.info("XGBOOST\naverage f1 : %.2f\naverage positive F1 :%.2f" % (
        np.average(np.array(total_scores["xgboost"]["f1_macro"])),
        np.average(np.array(total_scores["xgboost"]["f1_positive"]))))
    logging.info("SVM\naverage f1 : %.2f\naverage positive F1 :%.2f" % (
        np.average(np.array(total_scores["svm"]["f1_macro"])),
        np.average(np.array(total_scores["svm"]["f1_positive"]))))
    logging.info("LOGREGRESSION\naverage f1 : %.2f\naverage positive F1 :%.2f" % (
        np.average(np.array(total_scores["logisticRegression"]["f1_macro"])),
        np.average(np.array(total_scores["logisticRegression"]["f1_positive"]))))

    total_scores = {"f1_macro": [], "f1_positive": []}
    counter = 0
    kf = StratifiedKFold(n_splits=5)
    logging.info("#########.............. moderation for quality.....################")
    for train_index, test_index in kf.split(dataforqual, dataforqual["MODFORQUAL"]):
        train, test = dataforqual.iloc[train_index], dataforqual.iloc[test_index]
        train_x, test_x = train.drop("MODFORQUAL", axis=1), test.drop("MODFORQUAL", axis=1)
        # train_x, test_x = train.drop("MODINTERVENTION", axis=1), test.drop("MODINTERVENTION", axis=1)

        # scale features
        scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        train_y, test_y = train["MODFORQUAL"], test["MODFORQUAL"]
        logging.info("###XGBOOST###")

        report = xg_boost(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)
        total_scores["xgboost"]["f1_macro"].append(report["macro avg"]["f1-score"])
        total_scores["xgboost"]["f1_positive"].append(report["1"]["f1-score"])
        logging.info("F1 macro on split %d: %s" % (counter, report["macro avg"]["f1-score"]))
        logging.info("F1 positive on split %d: %s" % (counter, report["1"]["f1-score"]))

        logging.info("###SVM###")

        report = svm(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)

        total_scores["svm"]["f1_macro"].append(report["macro avg"]["f1-score"])
        total_scores["svm"]["f1_positive"].append(report["1"]["f1-score"])
        logging.info("F1 macro on split %d: %s" % (counter, report["macro avg"]["f1-score"]))
        logging.info("F1 positive on split %d: %s" % (counter, report["1"]["f1-score"]))
        logging.info("###logistic Regression###")

        report = logistic_regression(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)

        total_scores["logisticRegression"]["f1_macro"].append(report["macro avg"]["f1-score"])
        total_scores["logisticRegression"]["f1_positive"].append(report["1"]["f1-score"])

        logging.info("F1 macro on split %d: %s" % (counter, report["macro avg"]["f1-score"]))
        logging.info("F1 positive on split %d: %s" % (counter, report["1"]["f1-score"]))
    logging.info("----------------")
    logging.info("results test set")
    """
    logging.info("XGBOOST\naverage f1 : %.2f\naverage positive F1 :%.2f" % (
        np.average(np.array(total_scores["xgboost"]["f1_macro"])),
        np.average(np.array(total_scores["xgboost"]["f1_positive"]))))
    """
    logging.info("SVM\naverage f1 : %.2f\naverage positive F1 :%.2f" % (
        np.average(np.array(total_scores["svm"]["f1_macro"])),
        np.average(np.array(total_scores["svm"]["f1_positive"]))))
    logging.info("LOGREGRESSION\naverage f1 : %.2f\naverage positive F1 :%.2f" % (
        np.average(np.array(total_scores["logisticRegression"]["f1_macro"])),
        np.average(np.array(total_scores["logisticRegression"]["f1_positive"]))))
    counter += 1
