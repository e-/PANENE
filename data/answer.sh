#!/usr/bin/env bash

. ./metadata.sh

name=$1
dim="${dims[$name]}"
train_n="${train_ns[$name]}"

for version in $versions
do
  echo "Working on ${name}.${version}"
  args="${name}/${name}.${version}.bin ${name}/test.bin ${name}/${name}.${version}.answer.txt ${train_n} ${test_n} ${dim} ${k}"
  
  echo ./answer $args
  ./answer $args
done
