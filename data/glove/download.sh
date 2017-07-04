echo "Downloading GLOVE..."
wget "https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz"
echo "Extracting GLOVE..."
gzip -d glove.twitter.27B.100d.txt.gz
echo "Converting GLOVE..."
cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt # strip first column
python2 ../binary_converter.py glove.txt data.bin
python2 ../binary_converter.py --sample data.bin train.bin test.bin $TEST_N 100
python2 preprocess.py data.bin 1000000 100 glove

if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm glove.txt
    rm glove.twitter.27B.*
fi

