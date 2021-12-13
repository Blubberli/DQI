from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from data import EuropolisSimpleDataset
from evaluation import average_all, average_class


def get_datasets(data_dir, quality_dim, text_col, i):
    train = EuropolisSimpleDataset(path_to_dataset=data_dir + "/split%i/train.csv" % i,
                                   label=quality_dim, text_col=text_col)
    val = EuropolisSimpleDataset(path_to_dataset=data_dir + "/split%i/val.csv" % i,
                                 label=quality_dim, text_col=text_col)
    test = EuropolisSimpleDataset(path_to_dataset=data_dir + "/split%i/test.csv" % i,
                                  label=quality_dim, text_col=text_col)
    return train, val, test


def majority_baseline(train, test, label):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train, list(train[label]))
    test_y = list(test[label])
    predictions = dummy_clf.predict(test)
    # print(classification_report(y_true=test_y, y_pred=predictions))
    report = classification_report(y_true=test_y, y_pred=predictions, output_dict=True)
    return report


if __name__ == '__main__':
    import pandas as pd

    test_reports = []
    for i in range(0, 5):
        train, dev, test = get_datasets(data_dir="data/5foldStratified/int1", quality_dim="int1",
                                        text_col="cleaned_comment", i=i)
        test_report = majority_baseline(train.dataset, test.dataset, "int1")
        print(test_report)
        test_df = pd.DataFrame.from_dict(test_report, orient="index",
                                         columns=["precision", "recall", "f1-score", "support"])
        print(test_df.columns)
        test_reports.append(test_report)
    # print(average_all(test_reports))
    label_list = list(set(train.labels))
    label_list = [str(l) for l in label_list]
    print(average_class(test_reports, label_list=label_list))
    # print(test_reports[0])
    # print(test_reports[0].keys())
