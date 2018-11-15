import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier



D = datasets.load_iris()
data = D['data']
y = D['target']

pca = PCA(n_components=2)
X = pca.fit_transform(data)

e, vecs = np.linalg.eig(np.cov(np.transpose(data)))
e = np.sort(e)[::-1]

clf = SVC(kernel='poly', coef0=1.)
clf.fit(X,y)

colors = ['r', 'g', 'b']
marks = '^+x'
rgb = ListedColormap(colors)


h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)


plt.figure()
plt.title("Projection of Iris Dataset onto 2 Principal Components")



plt.pcolormesh(xx, yy, Z, cmap=rgb, alpha=0.1, shading='gouraud')

for i in range(3):
    indices = np.where(y == i)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[i], marker=marks[i], label=D['target_names'][i])

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.xlabel("First Principal Component $\lambda_1 = %0.2f$"%e[0])
plt.ylabel("Second Principal Component $\lambda_2 = %0.2f$"%e[1])
plt.show()

