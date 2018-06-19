import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pykalman import KalmanFilter
import random

fig = plt.figure(figsize=(16, 6))
ax = fig.gca()
plt.ioff()


def plota():
    rnd = random.randint(0,1000)
    grava = open("kalman.csv", "a")
    grava.write(str(rnd)+"\n")
    grava.close()
    df = pd.read_csv("kalman.csv")
    x = df.bids[-100:].dropna()
    cm = x[0:1]

    cm_seq = np.arange(1,cm, step=150)
    cm_lis = np.asarray(cm_seq)
    cm_com = cm_lis.tolist()
    cm_com = cm_com  + np.asarray(x).tolist()

    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=.09 * np.eye(2))
    states_pred = kf.em(cm_com).smooth(cm_com)[0]

    kf2 = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=.9 * np.eye(2))
    states_pred2 = kf2.em(cm_com).smooth(cm_com)[0]

    ax.clear()
    ax.plot(cm_com, ':', label="Random ") # + str(np.around(float(x[-1:]))))
    ax.plot( 	states_pred[:, 0], label="Kalman Filter 0.09") # + str(np.around(float(states_pred[:, 0][-1:]))))
    ax.plot(states_pred2[:, 0], label="Kalman Filter 0.9")# + str(np.around(float(states_pred[:, 0][-1:]))))

    # ax.plot(states_pred2[:, 0], label = "Kalman Filter2 " + str(np.around(float(states_pred[:, 0][-1:]))))
    # ax.plot(np.asarray(rm_), label = "Rolling Mean", alpha = .4)
    # l = rm_ - std
    # ax.plot(np.asarray(l), label = "Standard Moving", alpha = .4)
    #ax.set_xlim(40,len(states_pred[:]))
    #ax.set_ylim(float(states_pred[:, 0][-1:]) - 20, float(states_pred[:, 0][-1:]) + 20)
    ax.legend()


while (True):
    plt.pause(0.001)
    plt.ion()
    plota()
