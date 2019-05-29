# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
#from pyod.models.knn import KNN
#from pyod.utils import evaluate_print
#from pyod.models.so_gaal import SO_GAAL
#from pyod.models.mo_gaal import MO_GAAL
import matplotlib.pyplot as plt

def get_data():
    x = []
    y = []
    anomalous_num = []
    window_size = 120
    for i in range(1, 68):
        x_i = []
        y_i = []
        path_i = "/home/jerrry/Docments/ComputerNetwork/FinancialCrisisAL/code/A1Benchmark/real_" + str(i) + ".csv"
        tmp = pd.read_csv(path_i)
        minx = tmp["value"].min()
        maxx = tmp["value"].max()
        norm = tmp["value"].apply(lambda x: float(x - minx) / (maxx - minx))
        tmp = tmp.drop("value", axis=1)
        tmp["value"] = norm
        for j in range(tmp.shape[0]):
            features = []
            features.append(tmp["value"][j])
            x.append(features)
            x_i.append(features)
            y.append(tmp["is_anomaly"][j])
            y_i.append(tmp["is_anomaly"][j])
        x_i = np.array(x_i)
        y_i = np.array(y_i)
        x_i = x_i.reshape(y_i.shape)
        y_anomaly = np.where(y_i==1)
        x_dot = np.reshape(x_i, [-1, 1])
        anom_value = x_dot[y_anomaly]
        """
        plt.subplot(121)
        plt.plot(range(len(x_i)), x_i)
        plt.title("Data Distribution")
        plt.tight_layout()

        plt.subplot(122)
        plt.scatter(range(len(x_i)), x_i)
        plt.scatter(y_anomaly, anom_value, c='r')
        plt.title("Data Marked With Anomalies")
        plt.show()
        """
    #x = np.array(x)
    #y = np.array(y)
        #np.save("data/A1X_" + str(i) + ".npy", x_i)
        #np.save("data/A1Y_" + str(i) + ".npy", y_i)


def knn_detector():
    for i in range(1, 68):
        x = np.load("A1X_" + str(i) + ".npy")
        y = np.load("A1Y_" + str(i) + ".npy")
        clf_name = "KNN"
        clf = KNN()
        clf.fit(x)
        y_pred = clf.labels_
        y_scores = clf.decision_scores_
        evaluate_print(clf_name, y, y_scores)

def SO_GAAL_detector():
    x = np.load("A1X.npy")
    y = np.load("A1Y.npy")
    clf_name = "SO_GAAL"
    clf = SO_GAAL()
    #clf.fit_predict_score(x, y, scoring="roc_auc_score")
    clf.fit_predict_score(x, y, scoring="prc_n_score")

def MO_GAAL_detector():
    x = np.load("A1X.npy")
    y = np.load("A1Y.npy")
    clf_name = "SO_GAAL"
    clf = MO_GAAL()
    #clf.fit_predict_score(x, y, scoring="roc_auc_score")
    clf.fit_predict_score(x, y, scoring="prc_n_score")


if __name__=="__main__":
    get_data()
    #get_all_normal_data()
    #knn_detector()
    #SO_GAAL_detector()
    #MO_GAAL_detector()



# KNN: ROC: 0.6026; precision: 0.0377
# SO_GAAL: ROC: 0.3993; precision: 0.0024
# MO_GAAL: ROC: 0.4427; precision: 0.0222
