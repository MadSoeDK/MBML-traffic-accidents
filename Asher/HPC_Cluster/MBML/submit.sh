#!/bin/bash
#BSUB -J MBML
#BSUB -q hpc
#BSUB -W 120
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -n 1
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o Outputs/Jobout/MBML_1_%J.out
#BSUB -e Outputs/Jobout/MBML_1_%J.err

# Initialize Python environment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

TMP_VENV_314=$(mktemp -d)
python3 -m venv $TMP_VENV_314
source $TMP_VENV_314/bin/activate

# Install packages
pip install --no-cache-dir geopandas pyro-ppl torch contextily shapely pandas

python LGCP.py

deactivate
rm -rf $TMP_VENV_314