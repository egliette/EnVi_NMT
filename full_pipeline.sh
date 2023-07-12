#!/bin/bash

config_fpath="config.yml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            config_fpath="$2"
            shift
            ;;
        *)
            # Ignore unrecognized flags or arguments
            ;;
    esac
    shift
done

echo "---PREPROCESSING---"
python preprocess.py --config $config_fpath

echo "---TRAINING---"
python train.py --config $config_fpath

echo "---TESTING---"
python test.py --config $config_fpath