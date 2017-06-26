#!/bin/env python3

import csv
import os
from collections import defaultdict
import numpy as np

def main():
    count = defaultdict(lambda: defaultdict(int))

    dim1 = 5
    dim2 = 6
    bins = 200

    with open(os.path.join('..', '..', 'data', 'glove.shuffled.txt')) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        xs = []
        ys = []
        for row in reader:
            xs.append(float(row[dim1]))
            ys.append(float(row[dim2]))
        
        count, xedges, yedges = np.histogram2d(xs, ys, bins=bins)

    print(xedges)
    print(yedges)
            
    with open('baseline.csv', 'w') as outfile:
        print('x,y,count', file = outfile)

        for x, row in enumerate(count):
            for y, count in enumerate(row):
                if count > 0:
                    print('{},{},{}'.format(x, y, int(count)), file = outfile)

    with open('data.txt', 'w') as outfile:
        for i in range(len(xs)):
            print('{} {}'.format(xs[i], ys[i]), file=outfile)

    with open('data.sorted.txt', 'w') as outfile:
        s = sorted(list(zip(xs, ys)), key=lambda x: x[0] + x[1])
        for i in range(len(xs)):
            print('{} {}'.format(s[i][0], s[i][1]), file=outfile)

if __name__ == '__main__':
    main()

