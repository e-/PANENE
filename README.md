PANENE
--

PANENE (Progressive Approximate k-NEarest NEighbors) is a novel algorithm for the k-nearest neighbor (KNN) problem. In contrast to previous algorithms such as [Annoy](https://github.com/spotify/annoy), [FLANN](http://www.cs.ubc.ca/research/flann/), or many others from [this benchmark](https://github.com/erikbern/ann-benchmarks#evaluated), PANENE is *progressive*: it can process multiple mini-batches *online* while keeping each iteration bounded in time.

PANENE is based on [the FLANN library](https://github.com/mariusmuja/flann) and currently under development. We are going to add the following features:

- [x] a progressive k-d tree
- [x] a KNN table structure that enables constant-time lookup of KNN neighbors
- [X] Python bindings


# Installation

The source code was compiled and tested on Ubuntu 16.04 and macOS. The instructions below assume that you are using Ubuntu 16.04. If you are using macOS, please use an appropriate package manager (e.g., `brew` instead of `apt`)

Prerequisites:
- A C++ compiler with OpenMP and C++ 11 support
- Python 3.6 (we recommend to use Anaconda)
- CMake

Download the source code:
```bash
git clone https://github.com/e-/PANENE.git
cd PANENE
```

Check if the system has CMake:
```bash
cmake

# if you don't have CMake, run the following command:
sudo apt install cmake
```

Check if the system has g++. Note that PANENE requires the OpenMP support. Most recent compilers support it by default:
```bash
g++

# if you don't have g++, run the following command:
sudo apt install g++
```

Build the source code:
```bash
mkdir build
cd build
cmake ..
make
```

Install PANENE:
```bash
sudo make install
```

Update the shared library links after installation:
```bash
sudo ldconfig
```

# Running the Benchmark

Download the glove dataset and generate training and testing data (from the root directory):
```bash
cd data
./download.sh glove
```

Compute the exact neighbors of the testing data (this can take a while):
```bash
cp ../build/benchmark/answer .
./answer.sh glove
```

Run one of benchmarks (`kd_tree_benchmark` or `knn_table_benchmark`):
```bash
cd ..
cd build/benchmark
./kd_tree_benchmark
cat log.tsv
```
