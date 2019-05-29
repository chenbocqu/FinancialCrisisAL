# -*- coding: utf-8 -*-
import numpy as np
import heapq
from tensorflow import keras as K
import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

"""
import donut
from donut import standardize_kpi
from donut import Donut
from tfsnippet.modules import Sequential
from donut import DonutTrainer, DonutPredictor

"""

window_size = 60

"""
with tf.variable_scope('model') as model_vs:
    model = Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=window_size,
        z_dims=5,
    )
"""


def get_scores_complete():
    ## This function scores the testing data based on the whole dataset
    results = []
    # Supervised Setting
    #values, labels = np.load("A1X.npy"), np.load("A1Y.npy")
    #values = values.reshape((values.shape[0], ))

    # Unsupervised Setting
    values = np.load("A1X.npy")
    values = values.reshape((values.shape[0], ))
    labels = np.zeros_like(values, dtype=np.int32)


    test_portion = 0.2
    test_n = int(len(values) * test_portion)
    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]

    # The following 2 lines are used for data with oversampling
    #train_values, train_labels = np.load("aug_data/A1X_" + str(data_idx) + ".npy"), np.load("aug_data/A1Y_" + str(data_idx) + ".npy")

    # The following 2 lines are called when we need data type to be all normal, should be used with unsupervised setting
    train_y_normal_idx = np.where(train_labels==0)[0]
    train_values = train_values[train_y_normal_idx]

    train_values, mean, std = standardize_kpi(train_values)
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

    trainer = DonutTrainer(model=model, model_vs=model_vs)
    predictor = DonutPredictor(model)

    train_missing = np.zeros(shape=train_values.shape)
    test_missing = np.zeros(shape=test_values.shape)

    with tf.Session().as_default():
        trainer.fit(train_values, train_labels, train_missing, mean, std)
        test_score = predictor.get_score(test_values, test_missing)
    results.append(test_score)
    results = np.array(results)
    np.save("scores_s_60.npy", results)


def get_scores():
    valid_dataset = 67
    results = []
    for data_idx in range(1, 68):
        print ("Training dataset " + str(data_idx) + "...")
        # The following data loader indicates supervised setting
        #values, labels = np.load("data/A1X_"+str(data_idx) + ".npy"), np.load("data/A1Y_" + str(data_idx) + ".npy")

        # The following data loader indicates unsupervised setting
        values = np.load("data/A1X_"+str(data_idx) + ".npy")
        labels = np.zeros_like(values, dtype=np.int32)

        values = values.reshape((values.shape[0], ))
        test_portion = 0.2
        test_n = int(len(values) * test_portion)
        train_values, test_values = values[:-test_n], values[-test_n:]
        train_labels, test_labels = labels[:-test_n], labels[-test_n:]

        # The following 2 lines are used for data with oversampling
        train_values, train_labels = np.load("aug_data/A1X_" + str(data_idx) + ".npy"), np.load("aug_data/A1Y_" + str(data_idx) + ".npy")

        # The following 2 lines are called when we need data type to be all normal
        #train_y_normal_idx = np.where(train_labels==0)[0]
        #train_values = train_values[train_y_normal_idx]

        train_values, mean, std = standardize_kpi(train_values)
        test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

        trainer = DonutTrainer(model=model, model_vs=model_vs)
        predictor = DonutPredictor(model)

        train_missing = np.zeros(shape=train_values.shape)
        test_missing = np.zeros(shape=test_values.shape)

        with tf.Session().as_default():
            trainer.fit(train_values, train_labels, train_missing, mean, std)
            test_score = predictor.get_score(test_values, test_missing)
        results.append(test_score)

    results = np.array(results)
    np.save("scores_u_augmentation.npy", results) # The saved path can change according to your preferences


def get_metrics_complete(path):
    test_score = np.load(path)
    test_score = test_score[0]
    threshold = 0
    values = np.load("A1X.npy")
    labels = np.load("A1Y.npy")
    test_portion = 0.2
    test_n = int(len(labels) * test_portion)
    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    threshold = np.sum(test_score)/test_score.shape[0]
    test_correct = np.zeros(len(test_score))
    for j in range(len(test_labels)-window_size+1):
        for k in range(window_size):
            if (test_labels[j+k] == 1):
                test_correct[j] = 1
                break

    # This is used for "threshold" detector (Detector 1 in report)
    predictions = (test_score<threshold).astype(np.int32)

    # Use the anomalous_num if taking advanced detection method (Detector 2 in report)
    #anomalous_num = np.where(test_correct==1)[0].shape[0]
    #predictions = np.zeros_like(test_correct)
    #predictions_idx = heapq.nsmallest(anomalous_num, range(len(test_score)), test_score.take)
    #predictions[predictions_idx] = 1

    precision = precision_score(test_correct, predictions, average="binary")
    recall = recall_score(test_correct, predictions, average="binary")
    fscore = f1_score(test_correct, predictions, average="binary")
    roc_auc = roc_auc_score(test_correct, test_score)

    print ("Precision:", precision)
    print ("Recall:", recall)
    print ("F1-Score:", fscore)
    print ("ROC_AUC:", roc_auc)



def get_metrics(path):
    precision = []
    recall = []
    fscore = []
    roc_auc = []
    results = []
    valid_dataset = 67
    test_scores = np.load(path)
    threshold = 0
    for i in range(67):
        values = np.load("data/A1X_" + str(i+1) + ".npy")
        labels = np.load("data/A1Y_" + str(i+1) + ".npy")
        test_portion = 0.2
        test_n = int(len(labels) * test_portion)
        train_values, test_values = values[:-test_n], values[-test_n:]
        train_labels, test_labels = labels[:-test_n], labels[-test_n:]
        test_score = test_scores[i]
        threshold = np.sum(test_score)/test_score.shape[0]
        test_correct = np.zeros(len(test_score))
        for j in range(len(test_labels)-window_size+1):
            for k in range(window_size):
                if (test_labels[j+k] == 1):
                    test_correct[j] = 1
                    break

        # This is used for "threshold" detector (Detector 1 in report)
        predictions = (test_score<threshold).astype(np.int32)

        # Use the anomalous_num if taking advanced detection method (Detector 2 in report)
        #anomalous_num = np.where(test_correct==1)[0].shape[0]
        #predictions = np.zeros_like(test_correct)
        #predictions_idx = heapq.nsmallest(anomalous_num, range(len(test_score)), test_score.take)
        #predictions[predictions_idx] = 1

        precision.append(precision_score(test_correct, predictions, average="binary"))
        recall.append(recall_score(test_correct, predictions, average="binary"))
        fscore.append(f1_score(test_correct, predictions, average="binary"))
        if (np.sum(test_correct==1) == 0 or np.sum(test_correct==1) == test_correct.shape[0]):
            roc_auc.append(0)
            valid_dataset -= 1
        else:
            roc_auc.append(roc_auc_score(test_correct, test_score))

        # This part is for augmenting the dataset with infrequent normal sampels
        """
        train_score = np.load("scores_on_trained_S.npy")
        train_correct = np.zeros(len(train_score[i]))
        for j in range(len(train_labels)-window_size+1):
            for k in range(window_size):
                if (train_labels[j+k] == 1):
                    train_correct[j] = 1
                    break
        a_n = np.where(train_correct==1)[0].shape[0]
        pred = np.zeros_like(train_correct)
        pred_idx = heapq.nsmallest(a_n, range(len(train_score[i])), train_score[i].take)
        pred[pred_idx] = 1
        augment_data(train_correct, train_values[:-119], pred, 50, "aug_data/A1X_"+str(i+1)+".npy", "aug_data/A1Y_"+str(i+1)+".npy")
        """
    precision = np.array(precision)
    recall = np.array(recall)
    fscore = np.array(fscore)
    roc_auc = np.array(roc_auc)
    #np.save("precision.npy", precision)
    #np.save("recall.npy", recall)
    #np.save("fscore.npy", fscore)
    #np.save("roc_auc.npy", roc_auc)
    print ("Precision:", float(np.sum(precision))/valid_dataset)
    print ("Recall:", float(np.sum(recall))/valid_dataset)
    print ("Fscore:", float(np.sum(fscore))/valid_dataset)
    print ("AUC_Score:", float(np.sum(roc_auc))/valid_dataset)


def augment_data(labels, values, predictions, num, output_path_values, output_path_labels):
    # This function oversamples infrequent normal sample
    # Infrequent normal samples are determined by: The samples that are incorrectly
    # identified as anomalies but are normal samples
    # The augmented data are generated by hand...
    # num is the parameter for oversampled sampls number
    infrequent_idx = []
    true_anomaly_idx = np.where(labels==1)[0]
    pred_anomaly_idx = np.where(predictions==1)[0]
    augmented_values = values
    augmented_labels = labels
    augment_idx = []
    for idx in pred_anomaly_idx:
        if idx not in true_anomaly_idx:
            infrequent_idx.append(idx)
    if (len(infrequent_idx) == 0):
        np.save(output_path_values, values)
        np.save(output_path_labels, labels)
    else:
        infrequent_values = values[infrequent_idx]

        augment_idx = np.random.choice(infrequent_idx, num)
        augment_idx.sort()
        for k in range(augment_idx.shape[0]-1):
            if (augment_idx[k] == augment_idx[k+1]):
                augment_idx[k+1] = augment_idx[k] + 1
        #print (values.shape)
        for i in range(num):
            value_tmp = values[augment_idx[i]] + np.random.normal(0, 0.01, 1)[0]
            augmented_values = np.insert(augmented_values, augment_idx[i], value_tmp)
            augmented_labels = np.insert(augmented_labels, augment_idx[i], 0)
        #print (np.where(augmented_labels==1)[0])
        #print (augmented_values.shape)
        np.save(output_path_values, augmented_values)
        np.save(output_path_labels, labels)

    # Plot the original and new augmented dataset
    aug_idx = np.where(augmented_labels == 1)[0]

    values = values.reshape(labels.shape)
    augmented_values = augmented_values.reshape(augmented_labels.shape)
    #print (augmented_values.shape, augmented_labels.shape)
    #np.save(output_path_values, augmented_values)
    #np.save(output_path_labels, augmented_labels)
    plt.scatter(range(len(augmented_labels)), augmented_values)
    plt.scatter(aug_idx, augmented_values[aug_idx], c='r')
    plt.scatter(augment_idx, augmented_values[augment_idx], c='g')
    plt.show()


if __name__=="__main__":
    #get_scores()
    #get_scores_complete()
    #get_metrics("scores_unsupervised.npy")
    get_metrics_complete("scores_s_60.npy")


## Supervised, taking 0 as threshold
# All data with window size 120:
# precision: 0.2366
# recall: 0.4979
# f1_score: 0.3208
# ROC_AUC: 0.3511


# Seperate datasets evaluated by threshold, with window size 120
# Precision: 0.4287
# Recall: 0.7758
# f1_score: 0.4859
# ROC_AUC: 0.2707


# Window size 20:
# Precision: 0.1371
# Recall: 0.7888
# f1_score: 0.2125
# ROC_AUC: 0.1834

# Window size 60:
# Precision: 0.2584
# Recall: 0.7835
# f1_score: 0.3457
# ROC_AUC: 0.2460





# Different window size, different methods, different detection rules (thresholds or the same number).
# Supervised and unsupervised, adding infrequent normal sampels (deal with low recall)
# Training with all normal data, or training with abnormal ones included




# Pretrain: Supervised; Including abnormal data:
