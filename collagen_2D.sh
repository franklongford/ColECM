#!/bin/bash

declare param_file_name="$1"
declare pos_file_name="$2"
declare output_file_name="$3"

python3 -B src/2D/main_2D.py $param_file_name $pos_file_name $output_file_name
