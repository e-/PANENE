#!/usr/bin/env python3

# "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"

withdim = 5

with open("creditcard.csv", "r") as inf:
    with open("creditcard.txt", "w") as ouf:
        count = 0
        for line in inf.readlines():
            count += 1
            if count == 1:
                continue

            print(" ".join(line.split(",")[1:28]), file=ouf)
            if count % 10000 == 0:
                print("{} lines have been written".format(count))

