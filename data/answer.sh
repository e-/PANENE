#!/usr/bin/env bash

DATASETS="sift gist glove"
VERSIONS="original shuffled sorted"
DIMS=(128 960 100)
K=20
TRAIN_N=1000000
TEST_N=10000

index=0
for dataset in $DATASETS
do
  for version in $VERSIONS
  do
    echo "Working on ${dataset}.${version}"
    args="${dataset}/${dataset}.${version}.bin ${dataset}/test.bin ${dataset}/${dataset}.${version}.answer.bin ${TRAIN_N} ${TEST_N} ${DIMS[${index}]} ${K}"
    ./answer $args
  done

  index=$(expr $index + 1)
done

