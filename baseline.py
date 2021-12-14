from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from data import EuropolisSimpleDataset, EuropolisFeatureDataset
from evaluation import average_all, average_class
from sklearn.ensemble import RandomForestClassifier


def get_dimple_datasets(data_dir, quality_dim, text_col, i):
    train = EuropolisSimpleDataset(path_to_dataset=data_dir + "/split%i/train.csv" % i,
                                   label=quality_dim, text_col=text_col)
    val = EuropolisSimpleDataset(path_to_dataset=data_dir + "/split%i/val.csv" % i,
                                 label=quality_dim, text_col=text_col)
    test = EuropolisSimpleDataset(path_to_dataset=data_dir + "/split%i/test.csv" % i,
                                  label=quality_dim, text_col=text_col)
    return train, val, test


def get_feature_datasets(data_dir, quality_dim, text_col, i):
    train = EuropolisFeatureDataset(path_to_dataset=data_dir + "/split%i/train.csv" % i,
                                    label=quality_dim, text_col=text_col)
    val = EuropolisFeatureDataset(path_to_dataset=data_dir + "/split%i/val.csv" % i,
                                  label=quality_dim, text_col=text_col)
    test = EuropolisFeatureDataset(path_to_dataset=data_dir + "/split%i/test.csv" % i,
                                   label=quality_dim, text_col=text_col)
    return train, val, test


def random_forest_with_features(train, test, label):
    train_x = train[["cogency", "effectiveness", "reasonableness", "overall"]]

    test_x = test[["cogency", "effectiveness", "reasonableness", "overall"]]
    train_y = train[label]
    test_y = test[label]
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    probabilities = classifier.predict_proba(test_x)

    importance = classifier.feature_importances_
    feature_dic = {}
    # summarize feature importance
    for i, v in enumerate(importance):
        feature_dic[list(train_x.columns)[i]] = v
    feature_dic = dict(sorted(feature_dic.items(), key=lambda item: item[1]))
    for k, v in feature_dic.items():
        print("feature : %s, importance: %.2f" % (k, v))
    forest_report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    print(forest_report)
    return forest_report


def random_forest_with_all_features(train, test, label):
    feature_cols = list(train.num_cols)
    train_x = train.dataset[feature_cols]

    test_x = test.dataset[feature_cols]
    train_y = train.dataset[label]
    test_y = test.dataset[label]
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    probabilities = classifier.predict_proba(test_x)

    importance = classifier.feature_importances_
    feature_dic = {}
    # summarize feature importance
    for i, v in enumerate(importance):
        feature_dic[list(train_x.columns)[i]] = v
    feature_dic = dict(sorted(feature_dic.items(), key=lambda item: item[1]))
    for k, v in feature_dic.items():
        print("feature : %s, importance: %.2f" % (k, v))
    forest_report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    print(forest_report)
    return forest_report


def majority_baseline(train, test, label):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train, list(train[label]))
    test_y = list(test[label])
    predictions = dummy_clf.predict(test)
    report = classification_report(y_true=test_y, y_pred=predictions, output_dict=True)
    return report


if __name__ == '__main__':

    test_reports = []
    val_reports = []
    for i in range(0, 5):
        train, dev, test = get_feature_datasets(data_dir="data/5foldWithFeatures/int1", quality_dim="int1",
                                                text_col="cleaned_comment", i=i)
        test_report = random_forest_with_all_features(train, test, "int1")
        val_report = random_forest_with_all_features(train, dev, "int1")
        test_reports.append(test_report)
        val_reports.append(val_report)

    # print(average_all(test_reports))
    label_list = list(set(train.labels))
    label_list = [str(l) for l in label_list]
    print("average test")
    print(average_all(test_reports))
    print(average_class(test_reports, label_list=label_list))

    print("average dev")
    print(average_all(val_reports))

    print(average_class(val_reports, label_list=label_list))
