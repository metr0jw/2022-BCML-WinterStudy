# 2022 code by @metr0jw
# https://metr0jw.studio

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

dataset = load_iris()
data = dataset["data"][:, (2, 3)]
target = (dataset["target"] == 2).astype(np.float64)

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
params = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

for i, (gamma, C) in enumerate(params):
    rbf_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", gamma=gamma, C=C))
    ])
    rbf_svm.fit(data, target)

    # Visualize sample data
    plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.Paired)

    # Visualize Hyper-Plane
    decision_function = rbf_svm.decision_function(data)

    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = data[support_vector_indices]

    plt.subplot(2, 2, i + 1)
    plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = rbf_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C))
plt.tight_layout()
plt.savefig('./img/2a_rbf_kernel.png')
plt.show()
