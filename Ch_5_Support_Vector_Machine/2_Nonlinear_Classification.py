# 2022 code by @metr0jw
# https://metr0jw.studio

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

dataset = load_iris()
data = dataset["data"][:, (2, 3)]
target = (dataset["target"] == 2).astype(np.float64)

C = 10
degree = 3
kernel = ["Linear", "Polynomial", "RBF"]
classifier = [LinearSVC(C=C, loss="hinge"),
              SVC(kernel="poly", degree=degree, coef0=1, C=C),
              SVC(kernel="rbf", gamma=5, C=0.001)]

# Compare linear, poly, rbf kernel
for i in range(3):
    polynomial_svm_pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree)),
        ("scaler", StandardScaler()),
        ("svm", classifier[i])
    ])

    polynomial_svm_pipeline.fit(data, target)

    # Prediction
    print(polynomial_svm_pipeline.predict([[5.5, 1.7]]))

    # Visualize sample data
    plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.Paired)

    # Visualize Hyper-Plane
    decision_function = polynomial_svm_pipeline.decision_function(data)

    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = data[support_vector_indices]

    plt.subplot(3, 1, i+1)
    plt.scatter(data[:, 0], data[:, 1], c=target, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = polynomial_svm_pipeline.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")
    plt.title("Kernel: " + kernel[i])
plt.tight_layout()
plt.show()
