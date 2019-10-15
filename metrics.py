import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc


def all_metrics(yscore, y):
    """
        Inputs:
            yhat: binary predictions matrix
            y: binary ground truth matrix
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "pre", "rec", "f1"]
    yhat = np.round(yscore)

    # macro
    macro = all_macro(yhat, y)
    metrics = {names[i] + "_macro": macro[i] for i in range(len(macro))}
    metrics['auc_macro'] = mac_auc_score(y, yscore)

    # micro
    ymic = y.ravel()
    yscoremic = yscore.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)
    metrics.update({names[i] + "_micro": micro[i] for i in range(len(micro))})
    metrics['auc_micro'] = roc_auc_score(ymic, yscoremic)

    # mic auc-pr
    precision, recall, _ = precision_recall_curve(ymic, yscoremic)
    metrics['auc_pr'] = auc(recall, precision)
    return metrics


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic,
                                                                                                                ymic)


#########################################################################
# MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


##########################################################################
# MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (yhatmic.sum(axis=0) + 1e-10)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


def micro_metrics(yhatmic, ymic):
    tps = 0.     # true positives
    pred_pos = 0.   # predicted as positive (for precision)
    real_pos = 0.     # labeled as positive (for recall)
    num_classes = yhatmic.shape[1]

    for c in range(num_classes):
        yhat_c = yhatmic[:, c]
        y_c = ymic[:, c]
        tps += np.logical_and(yhat_c, y_c).sum()
        pred_pos += yhat_c.sum()
        real_pos += y_c.sum()

    precision = tps / (pred_pos + 1e-10)
    recall = tps / real_pos

    if precision + recall == 0:
        f1 = 0.
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = np.mean(yhatmic.flatten() == ymic.flatten())
    return accuracy, precision, recall, f1


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def mac_auc_score(y_true, y_score):
    num_classes = y_true.shape[1]
    aucs = []
    for c in range(num_classes):
        y_c = y_true[:, c]  # skip if no positive classes
        if y_c.sum() == 0:
            continue
        auc = roc_auc_score(y_c, y_score[:, c])
        aucs.append(auc)
    return np.mean(aucs)
