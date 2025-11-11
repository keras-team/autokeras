#!/bin/bash

SUCCESS_FILE="success.txt"
FAILURE_FILE="failure.txt"

# Clear the files
> $SUCCESS_FILE
> $FAILURE_FILE

for file in py/*.py; do
    if python3 "$file" > /dev/null 2>&1; then
        echo "$file" >> $SUCCESS_FILE
    else
        echo "$file" >> $FAILURE_FILE
    fi
done