echo "Downloading SIFT..."
wget "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz" -O bigann_learn.bvecs.gz
echo "Extracting SIFT..."
gunzip bigann_learn.bvecs.gz
echo "Converting SIFT..."
python2 ../binary_converter.py bigann_learn.bvecs data.bin 2500000
python2 ../binary_converter.py --sample data.bin train.bin test.bin $TEST_N 128
if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm bigann_learn.bvecs
fi
