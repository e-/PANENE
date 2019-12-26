#!/usr/bin/env python
import os
import array
import struct
import numpy as np

def mnist_image_to_text(fname, out):
    with open(fname, 'rb') as train_image:
        _, n, img_row, img_col = struct.unpack('>IIII', train_image.read(16))
        images = np.fromfile(train_image, dtype=np.uint8).reshape(n, img_row * img_col)
        data = images.astype(np.float32)

    with open(out, 'w') as outfile:
        for i, row in enumerate(data):
            print(' '.join([str(f) for f in row]), file=outfile)

def mnist_label_to_text(fname, out):
    with open(fname, 'rb') as train_label:
        _, n = struct.unpack('>II', train_label.read(8))
        labels = np.fromfile(train_label, dtype=np.uint8).reshape(n)
        data = labels.astype(dtype=np.uint8)

    with open(out, 'w') as outfile:
        for i, row in enumerate(data):
            print(row, file=outfile)

if __name__ == '__main__':
    import sys

    inf, outf = sys.argv[1], sys.argv[2]
    if inf.endswith('idx3-ubyte'):
        mnist_image_to_text(inf, outf)
    elif inf.endswith('idx1-ubyte'):
        mnist_label_to_text(inf, outf)
