#!/usr/bin/env python

size = 300000
from random import randint

def save(path, data):
    with open(path, "w") as ouf:
        for row in data:
            print >> ouf, ' '.join([str(f) for f in row])

for name in ['sift', 'glove']:
    rows = []
    with open(name + ".txt") as inf:
        i = 0
        for line in inf:
            rows.append(map(float, line.strip().split(' ')))
            i+= 1
            if i >= size:
                break
    
    dim = len(rows[0])
    # trim
    
    save("%s.trim.txt" % name, rows)

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

