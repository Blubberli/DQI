from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from data import EuropolisSimpleDataset, EuropolisFeatureDataset, EuropolisHumanAQ
from evaluation import average_all, average_class
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prepare_europolis import text2new_label
from collections import Counter
from sklearn import preprocessing
from sklearn.utils import compute_class_weight
import random

automatic_aq = ["cogency", "effectiveness", "reasonableness", "overall"]
human_aq = ["cogency_human", "effectiveness_human", "reasonableness_human", "overall_human"]

feature_mapping = {
    'Word Count': "number of words",
    'negative_adjectives_component': "negative adjectives",
    'social_order_component': 'social order',
    'action_component': 'action verbs',
    'positive_adjectives_component': 'positive adjectives',
    'joy_component': 'joy adjectives',
    'affect_friends_and_family_component': 'affect/affiliation nouns',
    'fear_and_digust_component': 'fear/disgust',
    'politeness_component': 'politeness words',
    'polarity_nouns_component': 'polarity nouns',
    'polarity_verbs_component': 'polarity verbs',
    'virtue_adverbs_component': 'hostility/rectitude gain adverbs',
    'positive_nouns_component': 'positive nouns',
    'respect_component': 'respect nouns',
    'trust_verbs_component': 'trust/joy/positive verbs',
    'failure_component': 'power loss/failure verbs',
    'well_being_component': 'well-being words',
    'economy_component': 'economy words',
    'certainty_component': 'sureness nouns, quantity',
    'positive_verbs_component': 'positive verbs',
    'objects_component': 'objects',
    'mattr50_aw': 'type token ratio',
    'mtld_original_aw': 'TTR window-based',
    'hdd42_aw': 'lexical diversity',
    'MRC_Familiarity_AW': 'familiarity score',
    'MRC_Imageability_AW': 'imageability score',
    'Brysbaert_Concreteness_Combined_AW': 'concreteness score',
    'COCA_spoken_Range_AW': 'COCA Range norms',
    'COCA_spoken_Frequency_AW': 'COCA frequency norms',
    'COCA_spoken_Bigram_Frequency': '#bigrams spoken corpus',
    'COCA_spoken_bi_MI2': 'squared mutual information of bigrams',
    'McD_CD': 'Semantic variability contexts',
    'Sem_D': 'Co-occurrence probability',
    'All_AWL_Normed': 'academic words',
    'LD_Mean_RT': 'lexical decision reaction time',
    'LD_Mean_Accuracy': 'Average lexical decision accuracy',
    'WN_Mean_Accuracy': ' Average naming accuracy',
    'lsa_average_top_three_cosine': 'Average LSA cosine score',
    'content_poly': 'poylsemous content words',
    'hyper_verb_noun_Sav_Pav': 'hypernyms'
}


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


def get_humanAQ(data_dir, quality_dim, text_col, i):
    train = EuropolisHumanAQ(path_to_dataset=data_dir + "/split%i/train.csv" % i,
                             label=quality_dim, text_col=text_col)
    val = EuropolisHumanAQ(path_to_dataset=data_dir + "/split%i/val.csv" % i,
                           label=quality_dim, text_col=text_col)
    test = EuropolisHumanAQ(path_to_dataset=data_dir + "/split%i/test.csv" % i,
                            label=quality_dim, text_col=text_col)
    return train, val, test


def get_class_names(label):
    name2class = text2new_label[label]
    class2name = dict(zip(name2class.values(), name2class.keys()))
    if label != "resp_gr":
        class_names = [class2name[i] for i in range(len(class2name))]
    else:
        class_names = ["disrespectful", "implicit respect", "explicit respect"]
    return class_names


def random_forest_with_features(train, test, label, features):
    train_x = train[features]

    test_x = test[features]
    train_y = train[label]
    test_y = test[label]
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    probabilities = classifier.predict_proba(test_x)
    plot_feature_importance(classifier, train_x.columns)

    forest_report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    print(forest_report)
    return forest_report


def filter_features(dataset, add_AQ, add_linguistic):
    features_cols = []
    if add_linguistic:
        features_cols = [
            'Word Count', 'negative_adjectives_component', 'social_order_component', 'action_component',
            'positive_adjectives_component',
            'joy_component', 'affect_friends_and_family_component', 'fear_and_digust_component', 'politeness_component',
            'polarity_nouns_component', 'polarity_verbs_component', 'virtue_adverbs_component',
            'positive_nouns_component',
            'respect_component', 'trust_verbs_component', 'failure_component', 'well_being_component',
            'economy_component',
            'certainty_component', 'positive_verbs_component', 'objects_component', 'mattr50_aw', 'mtld_original_aw',
            'hdd42_aw', 'MRC_Familiarity_AW', 'MRC_Imageability_AW', 'Brysbaert_Concreteness_Combined_AW',
            'COCA_spoken_Range_AW', 'COCA_spoken_Frequency_AW', 'COCA_spoken_Bigram_Frequency', 'COCA_spoken_bi_MI2',
            'McD_CD', 'Sem_D', 'All_AWL_Normed', 'LD_Mean_RT', 'LD_Mean_Accuracy', 'WN_Mean_Accuracy',
            'lsa_average_top_three_cosine', 'content_poly', 'hyper_verb_noun_Sav_Pav']

    #
    if add_AQ:
        for el in human_aq:
            features_cols.append(el)
    # 'Name_GI', 'Nation_Lasswell', 'Polit_GI'
    dataset = dataset[features_cols]
    dataset.rename(feature_mapping, inplace=True, axis=1)
    print(dataset.columns)
    return dataset


def plot_feature_importance(cls, features):
    importances = cls.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def plot_shap_values(classifier, dataset, class_names, output):
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(dataset)
    f = plt.figure()
    shap.summary_plot(shap_values, dataset.values, plot_type="bar", class_names=class_names,
                      feature_names=dataset.columns, max_display=10)
    f.savefig("/Users/falkne/PycharmProjects/DQI/data/plots/shap_plots/%s.png" % output, bbox_inches='tight',
              dpi=600)

    for i in range(len(class_names)):
        f = plt.figure()
        shap.summary_plot(shap_values[i], dataset, max_display=10)
        f.savefig("/Users/falkne/PycharmProjects/DQI/data/plots/shap_plots/%s_%s.png" % (output, class_names[i]),
                  bbox_inches='tight', dpi=600)


def plot_shap_values_svm(classifier, dataset, class_names, output, test_set):
    import fastshap
    number_of_rows = dataset.shape[0]
    random_indices = np.random.choice(number_of_rows, size=200, replace=False)
    background_sample = dataset[random_indices, :]
    ke = fastshap.KernelExplainer(classifier.predict, background_sample)
    shap_values = ke.calculate_shap_values(test_set[:5], verbose=True)
    # explainer = shap.KernelExplainer(classifier.predict, background_sample)
    # shap_values = explainer.shap_values(test_set)
    f = plt.figure()
    shap.summary_plot(shap_values, dataset.values, plot_type="bar", class_names=class_names,
                      feature_names=dataset.columns)
    f.savefig("/Users/falkne/PycharmProjects/DQI/data/plots/shap_plots/%s.png" % output, bbox_inches='tight',
              dpi=600)
    """
    for i in range(len(class_names)):
        f = plt.figure()
        shap.summary_plot(shap_values[i], dataset)
        f.savefig("/Users/falkne/PycharmProjects/DQI/data/plots/shap_plots/%s_%s.png" % (output, class_names[i]),
                  bbox_inches='tight', dpi=600)
    """


def random_forest_with_all_features(train, test, label, filter_feats, i, add_AQ):
    feature_cols = list(train.num_cols)
    train_y = train.dataset[label]
    test_y = test.dataset[label]
    if filter_feats:
        train_x = filter_features(train.dataset, add_AQ)
        test_x = filter_features(test.dataset, add_AQ)
    else:
        train_x = train.dataset[feature_cols]
        test_x = test.dataset[feature_cols]
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    probabilities = classifier.predict_proba(test_x)

    # plot_feature_importance(classifier, train_x.columns)
    forest_report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    # plot_shap_values(classifier=classifier, class_names=get_class_names(label), dataset=train_x,
    #                 output="%s_all_classes_split%d" % (label, i))

    return forest_report


def majority_baseline(train, test, label):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train, list(train[label]))
    test_y = list(test[label])
    predictions = dummy_clf.predict(test)
    report = classification_report(y_true=test_y, y_pred=predictions, output_dict=True)
    return report


def svm(train, test, label, filter_feats, add_AQ):
    feature_cols = list(train.num_cols)
    train_y = train.dataset[label]
    test_y = test.dataset[label]
    if filter_feats:
        train_x = filter_features(train.dataset, add_AQ)
        test_x = filter_features(test.dataset, add_AQ)
    else:
        train_x = train.dataset[feature_cols]
        test_x = test.dataset[feature_cols]
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    classifier = SVC(class_weight="balanced", kernel='linear')
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)

    # plot_feature_importance(classifier, train_x.columns)
    report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    # plot_shap_values_svm(classifier=classifier, class_names=get_class_names(label), dataset=train_x, test_set=test_x,
    #                     output="%s_all_classes_split%d" % (label, i))

    return report


def logistic_regression(train, test, label, filter_feats, add_AQ):
    feature_cols = list(train.num_cols)
    train_y = train.dataset[label]
    test_y = test.dataset[label]
    if filter_feats:
        train_x = filter_features(train.dataset, add_AQ)
        test_x = filter_features(test.dataset, add_AQ)
    else:
        train_x = train.dataset[feature_cols]
        test_x = test.dataset[feature_cols]
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    classifier = LogisticRegression(class_weight="balanced")
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)

    # plot_feature_importance(classifier, train_x.columns)
    report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)

    return report


def lgbm(train, test, label, filter_feats, add_AQ):
    feature_cols = list(train.num_cols)
    train_y = train.dataset[label]
    print(Counter(train_y))
    test_y = test.dataset[label]
    if filter_feats:
        train_x = filter_features(train.dataset, add_AQ)
        test_x = filter_features(test.dataset, add_AQ)
    else:
        train_x = train.dataset[feature_cols]
        test_x = test.dataset[feature_cols]
    # classifier = LGBMClassifier(n_estimators=1000, n_jobs=1)
    # classifier  = LogisticRegression(max_iter=50, C=0.3)
    classifier = GradientBoostingClassifier()
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    return classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)


def xg_boost(train, test, label, filter_feats, add_AQ, add_feats, i):
    feature_cols = list(train.num_cols)
    train_y = train.dataset[label]
    test_y = test.dataset[label]
    if filter_feats:
        train_x = filter_features(train.dataset, add_AQ=add_AQ, add_linguistic=add_feats)
        test_x = filter_features(test.dataset, add_AQ=add_AQ, add_linguistic=add_feats)
    else:
        train_x = train.dataset[feature_cols]
        test_x = test.dataset[feature_cols]
    # scaler = preprocessing.StandardScaler().fit(train_x)
    # train_x = scaler.transform(train_x)
    # test_x = scaler.transform(test_x)
    weights = compute_class_weight(y=train_y, class_weight="balanced", classes=np.unique(train_y))
    classifier = XGBClassifier(objective="multi:softmax", num_class=3, n_estimators=100)
    weights_instance = [weights[i] for i in train_y]
    classifier.fit(train_x, train_y, sample_weight=weights_instance)
    y_pred = classifier.predict(test_x)
    # plot_shap_values(classifier=classifier, class_names=get_class_names(label), dataset=test_x,
    #                 output="%s_all_classes_split%d" % (label, i))
    print(classification_report(y_true=test_y, y_pred=y_pred))
    return classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)


if __name__ == '__main__':

    test_reports = []
    val_reports = []
    for i in range(0, 5):
        train, dev, test = get_feature_datasets(
            data_dir="/Users/falkne/PycharmProjects/DQI/data/withHumanAQ/original/resp_gr",
            quality_dim="resp_gr",
            text_col="cleaned_comment", i=i)

        # test_report = random_forest_with_all_features(train, test, "resp_gr", filter_feats=True, i=i, add_AQ=True)
        test_report = xg_boost(train, test, "resp_gr", filter_feats=True, add_AQ=False, add_feats=True, i=i)
        # test_report = svm(train, dev, "jlev", filter_feats=True, add_AQ=False)
        # val_report = svm(train, dev, "int1", filter_feats=True, add_AQ=False)
        # val_reports.append(val_report)
        test_reports.append(test_report)

    # print(average_all(test_reports))
    label_list = list(set(train.labels))
    label_list = [str(l) for l in label_list]
    print("average test")
    print(average_all(test_reports))
    print(average_class(test_reports, label_list=label_list))

    # print("average dev")
    # print(average_all(val_reports))

    # print(average_class(val_reports, label_list=label_list))
