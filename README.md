PANENE
--

PANENE (Progressive Approximate k-NEarest NEighbors) is a novel algorithm for the k-nearest neighbor (KNN) problem. In contrast to previous algorithms such as [Annoy](https://github.com/spotify/annoy), [FLANN](http://www.cs.ubc.ca/research/flann/), and many others from [Benchmark](https://github.com/erikbern/ann-benchmarks#evaluated), it is *progressive*: it can process multiple mini-batches *online* while keeping each iteration bounded in time.

PANENE is based on [the FLANN library](https://github.com/mariusmuja/flann) and currently under development. We are going to add the following features:

- [x] a progressive k-d tree
- [ ] a KNN table structure that enables constant-time lookup

# Run the Benchmark

The source code was tested on Ubuntu 16.04, macOS, and Windows 10 (with Visual Studio 2017). However, shell scripts in this instruction will not run on Windows so use PowerShell instead or change them appropriately.

Prerequisites:
- A C++ compiler with OpenMP and C++11 support
- Python 2.7
- CMake or Visual Studio 2017 on Windows

Download the source code:
```bash
git clone https://github.com/e-/PANENE.git
```

Build the source code:
```bash
cd PANENE
mkdir build
cd build
cmake ..
make
```

Download the glove dataset and generate training and testing data:
```bash
cd ..
cd data
./download.sh glove
```

Compute the exact neighbors of the testing data (this can take a while):
```bash
./answer.sh glove
```

Run the benchmark:
```bash
cd ..
cd build/tests
./benchmark
cat log.tsv
```
