#!/bin/bash

# Directory to move files to
destination="rosetta_predictions"

# Create the destination directory if it does not exist
mkdir -p $destination

# Iterate over .ddg and .txt files in the predictions directory
for file in predictions/*/*/*.ddg predictions/*/*/*.txt; do
    # Extract the last two parts of the filename
    base=$(basename "$file")
    dir=$(dirname "$file")
    lastdir=$(basename "$dir")
    #sec_lastdir=$(basename "$(dirname "$dir")")

    # Concatenate the last two parts of the filename
    newname="${lastdir}_${base}"

    # Move the file to the destination directory
    cp "$file" "${destination}/${newname}"
done
