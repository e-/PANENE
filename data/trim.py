#!/usr/bin/env python

size = 280000 #300000
query_size = 100 #10000

import random
from random import randint
import numpy as np
from sklearn.decomposition import PCA
import sys

def save(path, data):
    with open(path, "w") as ouf:
        for row in data:
            print >> ouf, ' '.join([str(f) for f in row])

for name in ['creditcard']: #glove', 'sift']:
    rows = []
    try:
        with open(name + ".txt") as inf:
            n = 0
            for line in inf.readlines()[:size + query_size]:
                rows.append(map(float, line.strip().split(' ')))

                n += 1
                if n % 10000 == 0:
                    print("{} lines loaded".format(n))

    except Exception as e:
        print(e)
        print('Did you run %s.sh?'.format(name))
        sys.exit(1)
    
    dim = len(rows[0])
    
    ind = range(len(rows))
    random.shuffle(ind)

    query = [rows[ind[i]] for i in ind[:query_size]]

    save("%s.query.txt" % name, query)
    
    rows = [rows[ind[i]] for i in ind[query_size:query_size + size]]
    
    # shuffle
    save("%s.shuffled.txt" % name, rows)
    
    pca = PCA(1)
        
    arr = np.array(rows)
    xnew = pca.fit_transform(arr)

    save("%s.sorted.txt" % name, map(lambda x: x[0], sorted(zip(rows, xnew), key=lambda x:x[1])))
    
    continue
    # sorted
    
    # calculate the mean and stdev

    sums = [0.0] * dim

    for row in rows:
        for i, v in enumerate(row):
            sums[i] += v

    means = [s / size for s in sums]

    var = [0.0] * dim

    for row in rows:
        for i, v in enumerate(row):
            var[i] += (v - means[i]) ** 2
    
    index = var.index(max(var))

    rows = sorted(rows, key=lambda row: row[index])

    save("%s.sorted.txt" % name, rows)

    # half sorted
    
    for i in xrange(size / 2): 
        a = randint(0, size - 1)
        b = randint(0, size - 1)

        rows[a], rows[b] = rows[b], rows[a]

    save("%s.halfsorted.txt" % name, rows)

