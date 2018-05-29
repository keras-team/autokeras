import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# x1 = np.linspace(0.0, 5.0)
# x2 = np.linspace(0.0, 2.0)
#
# y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
# y2 = np.cos(2 * np.pi * x2)

# plt.subplot(1, 2, 1)
x1 = [0.01, 0.1, 1, 10, 100]
y1 = [1 - 0.6712, 1 - 0.7345, 1 - 0.8067, 1 - 0.8264, 1 - 0.8174]
plt.figure(figsize=(7, 2.5))
plt.semilogx(x1, y1, 'o-', label=r'$\beta$')
plt.xlabel(r'$\beta$')
plt.ylabel('Error Rate')
plt.gca().set_ylim([0.15, 0.35])

plt.savefig('/Users/haifeng/p1.png', bbox_inches='tight', pad_inches=0)


plt.clf()
x2 = x1
y2 = [1 - 0.7252, 1 - 0.8137, 1 - 0.8396, 1 - 0.8208, 1 - 0.817]
# plt.subplot(1, 2, 2)
plt.semilogx(x2, y2, 'o-', label=r'$\lambda$')
plt.xlabel(r'$\lambda$')
plt.ylabel('Error Rate')
plt.gca().set_ylim([0.15, 0.35])
plt.subplots_adjust(hspace=0.4)

# plt.show()
plt.savefig('/Users/haifeng/p2.png', bbox_inches='tight', pad_inches=0)
