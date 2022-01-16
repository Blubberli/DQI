import numpy as np
from data import EuropolisFeatureDataset
from sklearn.metrics import classification_report, f1_score, r2_score
from sklearn.ensemble import RandomForestClassifier
#from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from transformers import AutoTokenizer, RobertaTokenizer


features_cols = [
    'Word Count', 'negative_adjectives_component', 'social_order_component', 'action_component',
    'positive_adjectives_component',
    'joy_component', 'affect_friends_and_family_component', 'fear_and_digust_component', 'politeness_component',
    'polarity_nouns_component', 'polarity_verbs_component', 'virtue_adverbs_component', 'positive_nouns_component',
    'respect_component', 'trust_verbs_component', 'failure_component', 'well_being_component', 'economy_component',
    'certainty_component', 'positive_verbs_component', 'objects_component', 'mattr50_aw', 'mtld_original_aw',
    'hdd42_aw', 'MRC_Familiarity_AW', 'MRC_Imageability_AW', 'Brysbaert_Concreteness_Combined_AW',
    'COCA_spoken_Range_AW', 'COCA_spoken_Frequency_AW', 'COCA_spoken_Bigram_Frequency', 'COCA_spoken_bi_MI2',
    'McD_CD', 'Sem_D', 'All_AWL_Normed', 'LD_Mean_RT', 'LD_Mean_Accuracy', 'WN_Mean_Accuracy',
    'lsa_average_top_three_cosine', 'content_poly', 'hyper_verb_noun_Sav_Pav', 'cogency', 'reasonableness',
    'overall', 'effectiveness']


def filter_features(dataset):
    features_cols = [
        'Word Count', 'negative_adjectives_component', 'social_order_component', 'action_component',
        'positive_adjectives_component',
        'joy_component', 'affect_friends_and_family_component', 'fear_and_digust_component', 'politeness_component',
        'polarity_nouns_component', 'polarity_verbs_component', 'virtue_adverbs_component', 'positive_nouns_component',
        'respect_component', 'trust_verbs_component', 'failure_component', 'well_being_component', 'economy_component',
        'certainty_component', 'positive_verbs_component', 'objects_component', 'mattr50_aw', 'mtld_original_aw',
        'hdd42_aw', 'MRC_Familiarity_AW', 'MRC_Imageability_AW', 'Brysbaert_Concreteness_Combined_AW',
        'COCA_spoken_Range_AW', 'COCA_spoken_Frequency_AW', 'COCA_spoken_Bigram_Frequency', 'COCA_spoken_bi_MI2',
        'McD_CD', 'Sem_D', 'All_AWL_Normed', 'LD_Mean_RT', 'LD_Mean_Accuracy', 'WN_Mean_Accuracy',
        'lsa_average_top_three_cosine', 'content_poly', 'hyper_verb_noun_Sav_Pav', 'cogency', 'reasonableness',
        'overall', 'effectiveness']
    # 'Name_GI', 'Nation_Lasswell', 'Polit_GI'
    dataset = dataset[features_cols]
    return dataset


def surrogate_tree(dataset, label):
    classifier = DecisionTreeClassifier()
    features = filter_features(dataset.dataset)
    outcome_var = dataset.dataset[label]
    classifier.fit(X=features, y=outcome_var)
    y_pred = classifier.predict(features)
    r2 = r2_score(y_true=outcome_var, y_pred=y_pred)
    print("explained variance: %.2f" % r2)


def tree_based(dataset, label, plot=False):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifier = DecisionTreeClassifier(max_depth=5)
    counter = 0
    feature_importance = defaultdict(float)
    for train, test in kfold.split(dataset.dataset, dataset.dataset[label]):
        features = filter_features(dataset.dataset)
        outcome_var = dataset.dataset[label]
        features_train_set, features_test_set = features.loc[train], features.loc[test]
        outcome_train_set, outcome_test_set = outcome_var.loc[train], outcome_var.loc[test]
        classifier.fit(X=features_train_set, y=outcome_train_set)
        y_pred = classifier.predict(features_test_set)
        importance = classifier.feature_importances_
        feature_dic = {}
        # summarize feature importance
        for i, v in enumerate(importance):
            feature_dic[list(features.columns)[i]] = v
        feature_dic = dict(sorted(feature_dic.items(), key=lambda item: item[1]))
        for k, v in feature_dic.items():
            feature_importance[k] += v
        print("r2 score : %2f" % r2_score(y_true=outcome_test_set, y_pred=y_pred))
        if plot:
            viz = dtreeviz(classifier, features_train_set, outcome_train_set, target_name="predicted_labels",
                           feature_names=features_cols, class_names=list(classifier.classes_))
            viz.save("/Users/falkne/PycharmProjects/DQI/data/plots/trees/%s_%d.svg" % (label, counter))
        counter += 1
    for k, v in feature_importance.items():
        print("feature : %s, importance: %.2f" % (k, v / 5))


def get_probs(DenseDataWithIndex):
    results = pd.read_csv("/Users/falkne/PycharmProjects/DQI/results/augmented_base/jlev_all.csv", sep="\t")
    predictions = results.predictions.values
    probs = np.array(
        [np.array([float(i.replace("[", "").replace("]", "").strip()) for i in el.split(",")]) for el in predictions])
    return probs


def create_confusion_matrix():
    respect = pd.read_csv("/Users/falkne/PycharmProjects/DQI/results/augmented_base/respect_all.csv", sep="\t")
    jlev = pd.read_csv("/Users/falkne/PycharmProjects/DQI/results/augmented_base/jlev_all.csv", sep="\t")
    jcon = pd.read_csv("/Users/falkne/PycharmProjects/DQI/results/augmented_base/jcon_all.csv", sep="\t")
    int = pd.read_csv("/Users/falkne/PycharmProjects/DQI/results/augmented_base/int_all.csv", sep="\t")
    print(respect.columns)
    respect_confusion = confusion_matrix(y_true=respect.resp_gr.values, y_pred=respect.predicted_labels.values)
    int_confusion = confusion_matrix(y_true=int.int1.values, y_pred=int.predicted_labels.values)
    jlev_confusion = confusion_matrix(y_true=jlev["jlev"].values, y_pred=jlev.predicted_labels.values)
    jcon_confusion = confusion_matrix(y_true=jcon["jcon"].values, y_pred=jcon.predicted_labels.values)
    print(respect_confusion)
    df_respect = pd.DataFrame(respect_confusion, index=["disrespectful", "implicit", "explicit"], columns=
    ["disrespectful", "implicit", "explicit"])
    print(df_respect)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_respect, annot=True, fmt='.0f')
    plt.savefig("results/plots/respect_confusion.png", dpi=600)

    df_int = pd.DataFrame(int_confusion, index=["no reference", "negative", "neutral", "positive"],
                          columns = ["no reference", "negative", "neutral", "positive"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_int, annot=True, fmt='.0f')
    plt.savefig("results/plots/int_confusion.png", dpi=600)

    df_jlev = pd.DataFrame(jlev_confusion, index=["no justification", "inferior", "qualified", "sophisticated"],
                          columns=["no justification", "inferior", "qualified", "sophisticated"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_jlev, annot=True, fmt='.0f')
    plt.savefig("results/plots/jlev_confusion.png", dpi=600)

    df_jcon = pd.DataFrame(jcon_confusion, index=["own country", "no reference", "common good"],
                           columns=["own country", "no reference", "common good"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_jcon, annot=True, fmt='.0f')
    plt.savefig("results/plots/jcon_confusion.png", dpi=600)


def compute_shap_roberta():

    # p = get_probs(results)
    results = pd.read_csv("/Users/falkne/PycharmProjects/DQI/results/augmented_base/jlev_all.csv", sep="\t")
    explainer = shap.KernelExplainer(get_probs(results), results, link="logit")

    # print(results.predictions.values)


if __name__ == '__main__':
    # result_dataset = EuropolisFeatureDataset(
    #    path_to_dataset="/Users/falkne/PycharmProjects/DQI/results/augmented_base/jlev_all.csv",
    #    label="jlev", text_col="cleaned_comment")
    # tree_based(result_dataset, "predicted_labels")
    # compute_shap_roberta()
    #create_confusion_matrix()

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tokenizer.padding = True
    tokenizer.max_len = 512
    tokenizer.truncation = True
    print(tokenizer.padding)
    print(tokenizer.max_len)
    print(tokenizer.truncation)
