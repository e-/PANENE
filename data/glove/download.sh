echo "Downloading GLOVE..."
wget "https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz"
echo "Extracting GLOVE..."
gzip -d glove.twitter.27B.100d.txt.gz
echo "Converting GLOVE..."
cut -d " " -f 2- glove.twitter.27B.100d.txt > glove.txt # strip first column

train_n=$1
dim=$2
test_n=$3
remove_downloaded=$4

echo "Converting the data into a binary format"
python2 ../binary_converter.py glove.txt data.bin

echo "Data sampling"
python2 ../binary_converter.py --sample data.bin train.bin test.bin $test_n $dim

echo "Data shuffling"
python2 ../preprocess.py data.bin $train_n $dim glove

if [ "$remove_downloaded" = true ]; then
    rm glove.txt
    rm glove.twitter.27B.*
fi

