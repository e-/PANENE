echo "Downloading MNIST..."
wget "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" -O train-images-idx3-ubyte.gz
echo "Extracting MNIST..."
gunzip train-images-idx3-ubyte.gz
echo "Converting MNIST..."
python ../binary_converter.py train-images-idx3-ubyte data.bin
python ../binary_converter.py --sample data.bin train.bin test.bin 10000 784
if [ "$REMOVE_DOWNLOADED" = true  ]; then
  rm train-images-idx3-ubyte
fi
