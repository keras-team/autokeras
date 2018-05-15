import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from autokeras import ImageClassifier

clf = ImageClassifier(searcher_type='bayesian', path='/Users/haifeng/cifar10_backup', resume=True)
searcher = clf.load_searcher()
kernel_matrix = searcher.gpr.kernel_matrix
history = searcher.history
diff = np.zeros(kernel_matrix.shape)
for i, item1 in enumerate(history):
    for j, item2 in enumerate(history):
        e1 = item1['accuracy']
        e2 = item2['accuracy']
        diff[i][j] = 1 - abs(e1 - e2)

m1 = np.zeros(kernel_matrix.shape)
m2 = np.zeros(kernel_matrix.shape)
m3 = np.zeros(kernel_matrix.shape)
for i in range(47):
    for j in range(47):
        m1[i][j] = np.where(np.array(sorted(kernel_matrix.flatten())) == kernel_matrix[i][j])[0][0]
        m2[i][j] = np.where(np.array(sorted(diff.flatten())) == diff[i][j])[0][0]
        m1[i][j] = norm.ppf(m1[i][j] / (47.0 * 47.0))
        m2[i][j] = norm.ppf(m2[i][j] / (47.0 * 47.0))
        m3[i][j] = abs(m1[i][j] - m2[i][j])

print(m1)
print(m2)
print(kernel_matrix.shape)
print(diff.shape)


plt.matshow(m1, cmap='bone')
plt.savefig('/Users/haifeng/m1.png', bbox_inches='tight', pad_inches=0)

plt.matshow(m2, cmap='bone')
plt.savefig('/Users/haifeng/m2.png', bbox_inches='tight', pad_inches=0)

plt.matshow(m3, cmap='bone')
plt.savefig('/Users/haifeng/m3.png', bbox_inches='tight', pad_inches=0)
