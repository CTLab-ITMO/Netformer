#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit 1

function install_random_regressions() {
    echo "Installing datasets..."
    mkdir -p data
    cd data || exit 1
    wget -O random_regressions.csv "https://www.dropbox.com/scl/fi/vka3aonbo9qlsh0mhtlvw/random_regressions.csv?rlkey=6ey79ntvfkb056kplnbmc5d6z&st=6fepsd26&dl=0"
    cd ..
    echo "Datasets installed."
}

function install_trained_models() {
    echo "Installing trained models..."
    mkdir -p data
    cd data || exit 1
    gdown --no-check-certificate --remaining-ok --folder https://drive.google.com/drive/folders/1p9O2YagGicceNlrCtFFm-4eWysdMHF8N?usp=sharing
    cd ..
    echo "Trained models installed."
}

if [ "$#" -eq 0 ]; then
    install_random_regressions
    install_trained_models
    exit 0
else
    for arg in "$@"; do
        case $arg in
        -r | --random_regressions)
            install_random_regressions
            ;;
        -m | --trained_models)
            install_trained_models
            ;;
        *)
            echo "Invalid argument: $arg"
            ;;
        esac
    done
fi
