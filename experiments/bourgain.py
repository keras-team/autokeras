#!/usr/bin/python
import math
import numpy as np


def dist(x,y):
    return math.sqrt(sum([(x[i]-y[i])**2 for i in range(max(len(x),len(y)))]))

def bourgain_embedding(X,dist):
    np.random.seed(123)
    X_emb = []
    R = range(len(X))
    n = len(X)
    k = int(math.ceil(math.log(n)/math.log(2)-1))
    T = int(math.ceil(math.log(n)))
    counter = 0
    for i in range(0,k+1):
        for t in range(T):
            S = np.random.choice(R,2**i)
            for j in R:
                x = X[j]
                d = min([ dist(x,X[s]) for s in S])
                counter += len(S)
                if i==0 and t==0:
                    X_emb.append([d])
                else:
                    X_emb[j].append(d)
    print(counter)
    return X_emb


if __name__ == "__main__":
    X = [ [(i+j)*(i+j+1)/2+j for j in range(0,10+1)] for i in range(0,50)]
    X_emb = bourgain_embedding(X,dist)
    l = []
    for x in range(len(X)):
        for y in range(len(X)):
            if x != y:
                d1 = dist(X[x],X[y])
                d2 = dist(X_emb[x],X_emb[y])
                l.append(d2*1.0/(d1*1.0))
                #print x,y,d1,d2
    print("distortion = %s" % (max(l)/min(l)))
    print("upper bound for distortion = %s " % math.log(len(X)))
    # print "\n".join([str(x) for x in X_emb])
    #print "\n".join([",".join([str(a) for a in X_emb[i]]) for i in range(len(X))])