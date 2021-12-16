#!/bin/bash
source ~/.bashrc
CONDA="${CMO_CONDA_PATH}/bin/conda"
CONDA_BASE=$(${CONDA} info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate Thesis

latexmk -r pythontex-latexmkrc -pdf --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error $1
