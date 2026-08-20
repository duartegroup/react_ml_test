#!/bin/bash

# The directory containing the files (change this as needed)
source_directory="$1"

# Ensure a directory is passed as an argument
if [ -z "$source_directory" ]; then
  echo "Usage: $0 <source_directory>"
  exit 1
fi

# Check if the provided source directory exists
if [ ! -d "$source_directory" ]; then
  echo "The provided directory '$source_directory' does not exist."
  exit 1
fi

# Loop through all files in the source directory (non-recursively)
for file in "$source_directory"/*; do
  if [ -f "$file" ]; then
    # Create symlink in the parent directory (current directory)
    ln -s "$file" "$(dirname "$source_directory")/$(basename "$file")"
    echo "Created symlink for: $(basename "$file")"
  fi
done

echo "Symlink creation complete."