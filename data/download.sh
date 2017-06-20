#!/usr/bin/env bash

#REMOVE_DOWNLOADED=true # remove downloaded datasets after they've been converted
TEST_N=100 # number of test queries
DATASETS="trevi stl10 mnist" #" sift glove gist"

export REMOVE_DOWNLOADED
export TEST_N

for dataset in $DATASETS
do
  echo "Working on $dataset"
  (cd $dataset && ./download.sh)
done

