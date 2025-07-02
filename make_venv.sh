#!/bin/bash

## This script installs the necessary packages to run MMSFM.
## Specific archived commits from Github are downloaded for the
## MIOFlow and torchcfm packages for consistency with our working environment

## First create venv and activate (optional, highly recommended)
## e.g.) >>> conda create -p ./mmsfmvenv python=3.10
##       >>> conda activate ./mmsfmvenv

download_commit_archive() {
    ## Utility function. Downloads a project at a specific commit.
    PROJECT=$1
    COMMIT=$2
    OUTNAME=$3

    wget -O "${OUTNAME}.zip" "https://github.com/${PROJECT}/archive/${COMMIT}.zip"
    unzip "${OUTNAME}.zip"
    mv "${OUTNAME}-${COMMIT}" "${OUTNAME}"
    rm "${OUTNAME}.zip"  ## cleanup
}

## If not exists, download MIOFlow from a specific commit
if [[ ! -d "MIOFlow" ]]; then
    MIOCOMMIT="1b09f2c7eefefcd75891d44bf86c00a4904a0b05"
    printf "MIOFlow not found. Downloading from Github...\n"
    printf "Commit hash: ${MIOCOMMIT}\n"
    download_commit_archive "KrishnaswamyLab/MIOFlow" $MIOCOMMIT "MIOFlow"
else
    printf "MIOFlow found. Skip downloading from Github.\n"
fi

## If not exists, download conditional-flow-matching from a specific commit
if [[ ! -d "conditional-flow-matching" ]]; then
    CFMCOMMIT="af8fec6f6dc3a0dc7f8fb25d2ee0ca819fa5412f"
    printf "conditional-flow-matching not found. Downloading from Github...\n"
    printf "Commit hash: ${CFMCOMMIT}\n"
    download_commit_archive "atong01/conditional-flow-matching" $CFMCOMMIT "conditional-flow-matching"
else
    printf "conditional-flow-matching found. Skip downloading from Github.\n"
fi

## Install
python -m pip install -r requirements.txt
python -m pip install -e ./conditional-flow-matching
python -m pip install -e ./MIOFlow
python -m pip install -e .

echo "Virtual environment setup complete."

