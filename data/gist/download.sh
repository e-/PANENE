TEST_N=100
echo "Downloading GIST..."
#wget "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" -O gist.tar.gz
echo "Extracting GIST..."
#tar xzf gist.tar.gz
echo "Converting GIST..."
python2 ../binary_converter.py gist/gist_base.fvecs data.bin
python2 ../binary_converter.py --sample data.bin train.bin test.bin $TEST_N 960
if [ "$REMOVE_DOWNLOADED" = true ]; then
  rm -r gist
  rm gist.tar.gz
fi
