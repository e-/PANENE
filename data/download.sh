#!/usr/bin/env bash

. ./metadata.sh

name=$1
dim="${dims[$name]}"
train_n="${train_ns[$name]}"

echo "Downloading $name with $dim dimensions"

(cd $name && ./download.sh $train_n $dim $test_n $remove_downloaded)

