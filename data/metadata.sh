#!/usr/bin/env bash

remove_downloaded=false 
test_n=10000 # number of test queries
datasets="glove sift" #" gist trevi stl10 mnist" 

declare -A dims
dims=(["glove"]="100" ["sift"]="128")

declare -A train_ns
train_ns=(["glove"]="1000000" ["sift"]="1000000")

versions="original shuffled"
k=20

