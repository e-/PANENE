#!/usr/bin/env python
import os
import array
import struct

def main(in_name, out_name):
    with open(in_name, 'rb') as inf:
        with open(out_name, 'w') as outf:
            k = struct.unpack('L', inf.read(8))[0]

            print(k, file=outf)
            size = os.path.getsize(in_name)

            n = int((size - 8) / 12 / k)

            for i in range(n):
                answer = struct.unpack('=' + 'qf' * k, inf.read(12 * k))
                print(' '.join(map(str, answer)), file=outf)

if __name__ == '__main__':
    import sys

    main(sys.argv[1], sys.argv[2])
