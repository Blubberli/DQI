import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import random

random.seed(41)
# merge the classes and give them meaningful names(?)
# plot the class distribution(?)
# have german, english, original language text, all preprocessed, have unique ID, have argQuality Scores
# create a stratified split for each quality dimension

label2textencoding = {
    "jlev": {
        0: 'no justification',
        1: 'inferior justification',
        2: 'qualified justification',
        3: 'sophisticated (broad)',
        4: 'sophisticated (depth)'

    },
    "jcon": {
        0: 'own country',
        1: 'no reference',
        2: 'reference to common good (EU)',
        3: 'reference to common good (solidarty)'

    },
    "resp_gr": {
        0: 'disrespectful',
        1: 'implicit respect',
        2: 'balanced respect',
        3: 'explicit respect'
    },
    "int1": {
        0: 'no reference',
        1: 'negative reference',
        2: 'neutral reference',
        3: 'positive reference'

    }
}

text2new_label = {
    "jlev": {
        'no justification': 0,
        'inferior justification': 1,
        'qualified justification': 2,
        'sophisticated (broad)': 3,
        'sophisticated (depth)': 3

    },
    "jcon": {
        'own country': 0,
        'no reference': 1,
        'reference to common good (EU)': 2,
        'reference to common good (solidarty)': 2

    },
    "resp_gr": {
        'disrespectful': 0,
        'implicit respect': 1,
        'balanced respect': None,
        'explicit respect': 2
    },
    "int1": {
        'no reference': 0,
        'negative reference': 1,
        'neutral reference': 2,
        'positive reference': 3

    }

}


def convert_label_column(dimension, label_list):
    """Convert the old labels to the new labels and return the new label list"""
    text_labels = [label2textencoding[dimension][l] for l in label_list]
    new_labels = [text2new_label[dimension][text_label] for text_label in text_labels]
    return new_labels


def create_europolis_with_merged_labels():
    """Create the new europolis dataframe with merged labels"""
    data = pd.read_csv("/Users/johannesfalk/PycharmProjects/DQIReReloaded/data/europolis_dqi_with_terciles.csv",
                       sep="\t")
    data["post_length"] = [len(comment.split(" ")) for comment in data["cleaned_comment"].values]
    data = data[data.post_length >= 15]
    print(len(data))
    data = data.rename(columns={"label": "jlev"})
    dimensions = ["jlev", "jcon", "resp_gr", "int1"]
    new_label_cols = []
    for dim in dimensions:
        label_col = data[dim].values
        new_label_col = convert_label_column(dimension=dim, label_list=label_col)
        new_label_cols.append(new_label_col)
    data = data.rename(columns={"jlev": "jlev_old_labels", "jcon": "jcon_old_labels", "resp_gr": "resp_gr_old_labels",
                                "int1": "int1_old_labels"})
    data["jlev"] = new_label_cols[0]
    data["jcon"] = new_label_cols[1]
    data["resp_gr"] = new_label_cols[2]
    data["int1"] = new_label_cols[3]
    data.to_csv("data/europolis_newDQI.csv", index=False, sep="\t")


def plot_class_distributions():
    """Read the whole dataset and plot label distribution for each quality dimension"""
    data = pd.read_csv("data/europolis_newDQI.csv", sep="\t")
    dimensions = ["jlev", "jcon", "resp_gr", "int1"]
    for dim in dimensions:
        counts = pd.crosstab(index=data[dim], columns='count')
        counts.plot.bar()
        plt.savefig(
            "/Users/johannesfalk/PycharmProjects/DQIReReloaded/plots/original_class_distribution/%s.png" % dim)


def create_stratified_split(quality_dim, output_dir):
    """Given a label and the output path, generate a 5fold (stratified) split"""
    data = pd.read_csv("data/europolis_newDQI.csv", sep="\t")
    # drop where label is None
    data = data[data[quality_dim].notna()]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    counter = 0
    for train, test in kfold.split(data, data[quality_dim]):
        train_set, test_set = data.loc[train], data.loc[test]
        train_set, val_set = train_test_split(train_set, test_size=0.25)

        train_set.to_csv("%s/split%d/train.csv" % (output_dir, counter),
                         sep="\t", index=False)
        val_set.to_csv("%s/split%d/val.csv" % (output_dir, counter), sep="\t",
                       index=False)
        test_set.to_csv("%s/split%d/test.csv" % (output_dir, counter),
                        sep="\t", index=False)
        counter += 1

        print("train: %d, val: %d, test:%d" % (len(train_set), len(val_set), len(test_set)))
        print('train: %.2f, val:%.2f, test: %.2f' % (
            len(train_set) / len(data), len(val_set) / len(data), len(test_set) / len(data)))


def check_respect():
    path = "/Users/johannesfalk/PycharmProjects/DQIReReloaded/data/5foldStratified/resp_gr"
    for i in range(0, 5):
        train = pd.read_csv(path + "/split%d/train.csv" % i, sep="\t")
        val = pd.read_csv(path + "/split%d/val.csv" % i, sep="\t")
        test = pd.read_csv(path + "/split%d/test.csv" % i, sep="\t")
        print(train["resp_gr"].values)

        labels = train["resp_gr"].values
        labels = [int(l) for l in labels]
        train["resp_gr"] = labels
        labels = val["resp_gr"].values
        labels = [int(l) for l in labels]
        val["resp_gr"] = labels
        labels = test["resp_gr"].values
        labels = [int(l) for l in labels]
        test["resp_gr"] = labels
        # train.to_csv(path + "/split%d/train.csv" %i, sep="\t", index=False)
        # val.to_csv(path + "/split%d/val.csv" %i, sep="\t", index=False)
        # test.to_csv(path + "/split%d/test.csv" %i, sep="\t", index=False)


def create_5fold_features():
    data = pd.read_csv("/Users/johannesfalk/PycharmProjects/DQIReReloaded/data/europolis_with_features.csv", sep="\t")
    dimensions = ["jlev", "jcon", "resp_gr", "int1"]
    dropped = False
    print(data.columns)
    for dim in dimensions:
        path = "data/5foldStratified/%s" % dim
        new_path = "data/5foldWithFeatures/%s" % dim
        for i in range(0, 5):
            train = pd.read_csv(path + "/split%d/train.csv" % i, sep="\t")
            val = pd.read_csv(path + "/split%d/val.csv" % i, sep="\t")
            test = pd.read_csv(path + "/split%d/test.csv" % i, sep="\t")
            if not dropped:
                to_drop = [el for el in train.columns if el != "ID"]
                data = data.drop(columns=to_drop)
                dropped = True

            train = pd.merge(train, data, on="ID")
            val = pd.merge(val, data, on="ID")
            test = pd.merge(test, data, on="ID")
            train.to_csv(new_path + "/split%d/train.csv" % i, sep="\t", index=False)
            val.to_csv(new_path + "/split%d/val.csv" % i, sep="\t", index=False)
            test.to_csv(new_path + "/split%d/test.csv" % i, sep="\t", index=False)


if __name__ == '__main__':
    # plot_class_distributions()
    #create_stratified_split("jlev", "data/5foldStratified/jlev")
    # check_respect()
    create_5fold_features()
