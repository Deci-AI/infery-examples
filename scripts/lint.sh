#!/usr/bin/env bash
# make sure to install the dev requirements before running this script.
# python3 -m pip install black "black[jupyter]"

[[ "$PWD" =~ inference-examples$ ]] && python3 -m black . && exit 0

echo "Please run this script from inference-examples root directory"
exit 1

