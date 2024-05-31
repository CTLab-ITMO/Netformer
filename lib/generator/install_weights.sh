#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR" || exit 1


function install_weights() {
    echo "Installing weights..."
    mkdir -p weights
    cd weights || exit 1
    gdown --no-check-certificate --remaining-ok --folder https://drive.google.com/drive/folders/1OaaluWVaDefyWX_UAw9loGPjS5QpDugP?usp=sharing
    cd ..
    echo "Weights installed."
}

if [ "$#" -eq 0 ]; then
    install_weights
    exit 0
else
    for arg in "$@"; do
        case $arg in
        -w | --weights)
            install_weights
            ;;
        *)
            echo "Invalid argument: $arg"
            ;;
        esac
    done
fi