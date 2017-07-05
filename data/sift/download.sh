echo "Downloading SIFT..."

cd "$(dirname "$0")"
wget "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
echo "Extracting SIFT..."
tar -xzf sift.tar.gz
echo "Converting SIFT..."
#python convert_texmex_fvec.py sift/sift_base.fvecs >> sift.txt
python2 ../binary_converter.py sift/sift_base.fvecs data.bin 2500000
python2 ../binary_converter.py --sample data.bin train.bin test.bin $TEST_N 128
python2 ../preprocess.py data.bin 1000000 128 sift

#wget "ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz" -O bigann_learn.bvecs.gz
#echo "Extracting SIFT..."
#gunzip bigann_learn.bvecs.gz
#echo "Converting SIFT..."
#python2 ../binary_converter.py bigann_learn.bvecs data.bin 2500000
#python2 ../binary_converter.py --sample data.bin train.bin test.bin $TEST_N 128
#python2 ../preprocess.py data.bin 1000000 128 sift

if [ "$REMOVE_DOWNLOADED" = true ]; then
#rm bigann_learn.bvecs
  rm -rf sift.tar.gz
  rm -rf sift
fi
