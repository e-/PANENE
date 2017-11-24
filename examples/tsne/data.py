#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import argparse
import sys
import struct

parser = argparse.ArgumentParser(description='Generate input data for the tsne example')

parser.add_argument('path', type=str, help='output path')
parser.add_argument('--sample', type=str, help='sample or random')
parser.add_argument('--text', dest='text', action='store_true', default=False, help='sample or random')
parser.add_argument('-n', type=int, default=10000, help='# of rows to write')
parser.add_argument('-d', type=int, default=100, help='# of dimensions to write')
parser.add_argument('--theta', '-t', type=float, default=0.5, help='theta')
parser.add_argument('--perplexity', '-p', type=float, default=10, help='target perplexity')
parser.add_argument('--output-dims', '-o', type=int, default=2, help='output dimensionality')
parser.add_argument('--max-iter', '-i', type=int, default=300, help='maximum # of iterations')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.sample:
        if args.text:
            with open(args.sample, 'r') as inf:
                with open(args.path, 'w') as outf:
                    print(args.n, file=outf)
                    print(args.d, file=outf)
                    print(args.theta, file=outf)
                    print(args.perplexity, file=outf)
                    print(args.output_dims, file=outf)
                    print(args.max_iter, file=outf)
                    
                    lines = inf.readlines()
                    for i in range(args.n):
                        print(lines[i], file=outf, end='')
        else:
            with open(args.sample, 'rb') as inf:
                with open(args.path, 'w') as outf:
                    print(args.n, file=outf)
                    print(args.d, file=outf)
                    print(args.theta, file=outf)
                    print(args.perplexity, file=outf)
                    print(args.output_dims, file=outf)
                    print(args.max_iter, file=outf)

                    for i in range(args.n):
                        inf.seek(i * 4 * args.d)
                        floats = struct.unpack('f' * args.d, inf.read(args.d * 4))
                        print(' '.join([str(f) for f in floats]), file=outf)
    else:
        r = random.random()

        with open(args.path, 'w') as outf:
            print(args.n, file=outf)
            print(args.d, file=outf)
            print(args.theta, file=outf)
            print(args.perplexity, file=outf)
            print(args.output_dims, file=outf)
            print(args.max_iter, file=outf)

            for i in range(args.n):
                print(' '.join([str(random.uniform(-1, 1)) for j in range(args.d)]), file=outf)
