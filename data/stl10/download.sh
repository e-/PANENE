echo "Downloading STL-10..."
#wget "http://cs.stanford.edu/~acoates/stl10/stl10_binary.tar.gz" -O stl10_binary.tar.gz
echo "Extracting STL-10..."
#tar xzf stl10_binary.tar.gz
echo "Converting STL-10..."
python2 ../binary_converter.py stl10_binary/unlabeled_X.bin data.bin
python2 ../binary_converter.py --sample data.bin train.bin test.bin $TEST_N 9216
python2 ../preprocess.py data.bin stl10 9216 gist

if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm -r stl10_binary
    rm stl10_binary.tar.gz
fi
