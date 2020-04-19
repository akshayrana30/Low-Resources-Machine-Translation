#!/bin/bash
#SBATCH  --time=01:00:00
#SBATCH  --mem=8G
#SBATCH  --gres=gpu:k80:1
#SBATCH  --output=./logfile
python evaluator.py --target-file-path ./test_fr.txt --input-file-path ./test_en.txt 

