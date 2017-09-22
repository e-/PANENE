#!/usr/bin/env python
import os
import array
import struct

def main(fname, num, dim, prefix = "data"):
    import random

    sz = os.path.getsize(fname)
    N = sz / (4 * dim)
    
    num = min(num, N)
    data = []

    with open(fname, 'rb') as inf:
        for i in range(num):
            if i % 1000 == 0:
                print(i, num)

            inf.seek(i * 4 * dim)
            floats = struct.unpack('f'*dim, inf.read(dim * 4))
            data.append(floats)

    indices = range(num)
    random.shuffle(indices)

    with open("{}.original.bin".format(prefix), 'wb') as outfile1:
        with open("{}.shuffled.bin".format(prefix), 'wb') as outfile2:
            for i in range(num):
                if i % 1000 == 0:
                    print(i, num)

                _write_floats(data[i], outfile1)
                _write_floats(data[indices[i]], outfile2)
    
def _write_floats(floats, outfile):
    float_arr = array.array('d', floats)
    s = struct.pack('f' * len(float_arr), *float_arr)
    outfile.write(s)


if __name__ == '__main__':
    import sys

    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
