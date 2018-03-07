#!/bin/bash

declare n_fibre="$1"
declare l_fibre="$2"
declare directory="$3"

python3 -B src/2D/main_2D.py $n_fibre $l_fibre $directory
