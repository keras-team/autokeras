#!/bin/bash
set -Euo pipefail

ruff check .
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  exit 1
fi

ruff format --check .
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
  exit 1
fi
for i in $(find autokeras benchmark -name '*.py') # or whatever other pattern...
do
  if ! grep -q Copyright $i
  then
    echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
  fi
done
