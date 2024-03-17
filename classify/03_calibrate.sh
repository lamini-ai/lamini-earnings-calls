#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build the container
$LOCAL_DIRECTORY/scripts/build.sh

docker run -it \
    -v ~/.lamini:/root/.lamini \
    -v $LOCAL_DIRECTORY/data:/app/lamini-earnings-classify/data \
    -v $LOCAL_DIRECTORY/models:/app/lamini-earnings-classify/models \
    --entrypoint /app/lamini-earnings-classify/scripts/calibrate.sh \
    lamini-earnings-classify:latest $@
