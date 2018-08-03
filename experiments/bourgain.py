#!/usr/bin/python
import math
import numpy as np


def dist(x, y):
    return math.sqrt(sum([(x[i] - y[i]) ** 2 for i in range(max(len(x), len(y)))]))


def bourgain_embedding(elements, dist):
    np.random.seed(123)
    distort_elements = []
    r = range(len(elements))
    n = len(elements)
    k = int(math.ceil(math.log(n) / math.log(2) - 1))
    t = int(math.ceil(math.log(n)))
    counter = 0
    for i in range(0, k + 1):
        for t in range(t):
            s = np.random.choice(r, 2 ** i)
            for j in r:
                x = elements[j]
                d = min([dist(x, elements[s]) for s in s])
                counter += len(s)
                if i == 0 and t == 0:
                    distort_elements.append([d])
                else:
                    distort_elements[j].append(d)
    return np.array(distort_elements)


if __name__ == "__main__":
    print(dist([1, 2], [3, 4]))
    X = [[(i + j) * (i + j + 1) / 2 + j for j in range(0, 10)] for i in range(0, 50)]
    print(X)
    X_emb = bourgain_embedding(X, dist)
    l = []
    for x in range(len(X)):
        for y in range(len(X)):
            if x != y:
                d1 = dist(X[x], X[y])
                d2 = dist(X_emb[x], X_emb[y])
                l.append(d2 * 1.0 / (d1 * 1.0))
                # print x,y,d1,d2
    print("distortion = %s" % (max(l) / min(l)))
    print("upper bound for distortion = %s " % math.log(len(X)))
    # print "\n".join([str(x) for x in X_emb])
    # print "\n".join([",".join([str(a) for a in X_emb[i]]) for i in range(len(X))])


def vector_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def bourgain_embedding_matrix(distance_matrix):
    np.random.seed(123)
    distort_elements = []
    r = range(len(distance_matrix))
    n = len(distance_matrix)
    k = int(math.ceil(math.log(n) / math.log(2) - 1))
    t = int(math.ceil(math.log(n)))
    counter = 0
    for i in range(0, k + 1):
        for t in range(t):
            s = np.random.choice(r, 2 ** i)
            for j in r:
                d = min([distance_matrix(j, s) for s in s])
                counter += len(s)
                if i == 0 and t == 0:
                    distort_elements.append([d])
                else:
                    distort_elements[j].append(d)
    distort_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distort_matrix[i][j] = distance_matrix[j][i] = vector_distance(distort_elements[i], distort_elements[j])
    return np.array(distort_elements)
