echo "Downloading SIFT..."

cd "$(dirname "$0")"
wget "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
echo "Extracting SIFT..."
tar -xzf sift.tar.gz
echo "Converting SIFT..."

train_n=$1
dim=$2
test_n=$3
remove_downloaded=$4

echo "Converting the data into a binary format"
python2 ../binary_converter.py sift/sift_base.fvecs train.bin $train_n

echo "Data sampling"
python2 ../binary_converter.py sift/sift_query.fvecs test.bin $test_n

echo "Data shuffling"
python2 ../shuffle.py train.bin $train_n $dim sift

if [ "$remove_downloaded" = true ]; then
  rm -rf bigann_learn.bvecs
  rm -rf sift.tar.gz
  rm -rf sift
fi
