#!/usr/bin/env bash

remove_downloaded=false 
test_n=1000 # number of test queries
datasets="sift glove gist trevi stl10 mnist" 

declare -A dims
dims=(["glove"]="100")

declare -A train_ns
train_ns=(["glove"]="1000000")

name=$1

dim="${dims[$name]}"
train_n="${train_ns[$name]}"

echo "Downloading $name with $dim dimensions"

(cd $name && ./download.sh $train_n $dim $test_n $remove_downloaded)

