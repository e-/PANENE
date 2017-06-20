echo "Downloading Trevi..."
wget "http://phototour.cs.washington.edu/patches/trevi.zip" -O trevi.zip
echo "Extracting Trevi..."
mkdir patches
unzip -q trevi.zip -d patches
echo "Converting Trevi..."
python2 ../binary_converter.py patches/ data.bin
python2 ../binary_converter.py --sample data.bin train.bin test.bin $TEST_N 4096
if [ "$REMOVE_DOWNLOADED" = true ]; then
    rm -r patches
    rm trevi.zip
fi
