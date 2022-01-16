import pandas as pd
import os
import re
from baseline import filter_features
from prepare_europolis import label2textencoding

def get_predicted_labels(predictions):
    """Given a list of string representations of prediction, e.g. '['[0.05, 0.95]']' convert the predictions into lists of floats and
    return a list with the predicted label (the index of the highest probability)
    """
    predictions = [[float(prob.strip()) for prob in el.replace("[", "").replace("]", "").split(",")] for el in
                   predictions]
    predicted_label = [el.index(max(el)) for el in predictions]
    return predicted_label


def merge_predictions_with_features(predictions_path, dimension, feature_path, output_path):
    prediction_files = os.listdir(predictions_path)
    pattern = re.compile("(fold)(\d)")
    all_test_sets = []
    for file in prediction_files:
        print(file)
        split = int(pattern.search(file).group(2))
        features = pd.read_csv("%s/%s/split%d/test.csv" % (feature_path, dimension, split), sep="\t")
        features = filter_features(features)
        predictions_frame = pd.read_csv("%s/%s" % (predictions_path, file), sep="\t")
        feature_columns = [el for el in features.columns if el not in predictions_frame.columns]
        features = features[feature_columns]
        merged = pd.concat([predictions_frame, features], axis=1)
        predicted_labels = get_predicted_labels(merged.predictions.values)
        merged["predicted_labels"] = predicted_labels
        merged["split"] = [split] * len(merged)
        all_test_sets.append(merged)
    all_predictions = pd.concat(all_test_sets)
    all_predictions.to_csv(output_path, index=False, sep="\t")

def extract_predictions_for_each_class(aggregated_frame, dim):
    aggregated_f = pd.read_csv(aggregated_frame, sep="\t")
    predictions = aggregated_f.predictions.values
    predictions = [[float(prob.strip()) for prob in el.replace("[", "").replace("]", "").split(",")] for el in
                   predictions]
    for i in range(len(predictions[0])):
        predictions_class_i = [el[i] for el in predictions]
        label_name = label2textencoding[dim][i]
        aggregated_f[label_name] = predictions_class_i
    aggregated_f.to_csv(aggregated_frame, sep="\t", index=False)




if __name__ == '__main__':
    extract_predictions_for_each_class("/Users/falkne/PycharmProjects/DQI/results/augmented_base/respect_all.csv", "resp_gr")
