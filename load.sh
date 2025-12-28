#!/bin/bash

set -e  # Exit on any error

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <ssh_key> <user@host>"
    exit 1
fi

KEY="$1"
SERVER="$2"
LOCAL="."
REMOTE="diagf"

# Check if local folder exists
[ -d "$LOCAL" ] || { echo "Error: $LOCAL not found"; exit 1; }

# Check if SSH key exists
[ -f ~/.ssh/"$KEY" ] || { echo "Error: SSH key ~/.ssh/$KEY not found"; exit 1; }

# Create remote folder
ssh -i ~/.ssh/"$KEY" "$SERVER" "mkdir -p ~/$REMOTE"

# Copy files
scp -r -i ~/.ssh/"$KEY" "$LOCAL"/src "$SERVER":~/"$REMOTE"/
scp -i ~/.ssh/"$KEY" "$LOCAL"/*.sh "$SERVER":~/"$REMOTE"/
scp -i ~/.ssh/"$KEY" "$LOCAL"/*.slurm "$SERVER":~/"$REMOTE"/
scp -i ~/.ssh/"$KEY" "$LOCAL"/b_vector.txt "$SERVER":~/"$REMOTE"/
scp -i ~/.ssh/"$KEY" "$LOCAL"/algebra.h "$SERVER":~/"$REMOTE"/
scp -i ~/.ssh/"$KEY" "$LOCAL"/CMakeLists.txt "$SERVER":~/"$REMOTE"/

echo "Done"