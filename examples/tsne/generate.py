import random

n = 10000
d = 50
theta = 0.3
perplexity = 5
no_dims = 2
max_iter = 200
r = random.random()

with open('data.dat', 'w') as inf:
    print(n, file=inf)
    print(d, file=inf)
    print(theta, file=inf)
    print(perplexity, file=inf)
    print(no_dims, file=inf)
    print(max_iter, file=inf)

    for i in range(n):
        for j in range(d):
            print(random.random(), file=inf, end=' ')
        print(file=inf)
