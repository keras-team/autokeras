#!/bin/bash

clear & clear

# First run all tests
for dir in $(find $PWD/* -maxdepth 0 -type d)
do
  cd $dir
  for uf in unittest_*.py
  do 
    echo Testing $uf %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    python $uf
  done
  cd ../
done

# Then clean all directories
for dir in $(find $PWD/* -maxdepth 0 -type d)
do
  echo Cleaning $dir
  cd $dir
  rm -rf *.py[co]
  rm -rf direct_log_*
  cd ../
done

echo Cleaning $PWD
rm -rf *.py[co]
