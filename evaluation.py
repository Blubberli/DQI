import statistics


def average_all(reports):
    """Get a string of the average of the global scores (Fmacro, Accuracy)"""
    s = "metric\tmean\tdeviation\n"
    for k in reports[0].keys():
        all_vals = [report[k] for report in reports]
        s += "%s\t%.2f\t%.2f\n" % (k, statistics.geometric_mean(all_vals), statistics.stdev(all_vals))
    return s


def average_class(reports):
    """Get a string of the average of the performance for each class"""
    s = "label\tmetric\tmean\tdeviation\n"
    for metric in reports[0]['0'].keys():
        for k in reports[0].keys():
            all_vals = [report[k][metric] for report in reports]
            s += "%s\t%s\t%.2f\t%.2f\n" % (metric, k, statistics.mean(all_vals), statistics.stdev(all_vals))
    return s
