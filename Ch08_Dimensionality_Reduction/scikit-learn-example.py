import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


dataset = load_wine()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
target = pd.DataFrame(dataset.target)
print(f'data: {data[:10]}')
print(f'target: {target[:10]}')

fig = plt.figure(figsize=(16,12))
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 30)
cax = ax1.imshow(data.corr(), interpolation="nearest", cmap=cmap)
ax1.grid(True)
plt.title('Wine data set features correlation\n',fontsize=15)
labels = data.columns
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_yticklabels(labels, fontsize=9)
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(data['od280/od315_of_diluted_wines'], data['flavanoids'], c=target.values, edgecolors='k',
            alpha=0.75, s=150)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation", fontsize=15)
plt.xlabel("OD280/OD315 of diluted wines",fontsize=15)
plt.ylabel("Flavanoids", fontsize=15)
plt.show()


pca = PCA(n_components=None)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', pca)
])

pipeline.fit(data)


plt.figure(figsize=(10, 6))
plt.scatter(x=[i+1 for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
            s=200, alpha=0.75, c='orange', edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n", fontsize=25)
plt.xlabel("Principal components", fontsize=15)
plt.xticks([i+1 for i in range(len(pca.explained_variance_ratio_))], fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio", fontsize=15)
plt.tight_layout()
plt.show()

X = pd.DataFrame(pipeline.fit_transform(data), columns=data.columns)

plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=target.values, edgecolors='k',
            alpha=0.75, s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n", fontsize=20)
plt.xlabel("Principal component-1", fontsize=15)
plt.ylabel("Principal component-2", fontsize=15)
plt.show()


