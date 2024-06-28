#!/bin/bash

source ./setup/setup_Linux.sh

clear

cd /bin/

python3 token_when.py --help

python3 token_when.py --load-mod
exit