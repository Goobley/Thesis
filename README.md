## My PhD Thesis

This repository contains the source for my [thesis](https://theses.gla.ac.uk/82584/).
All figures are built during LaTeX compilation using [pythontex](https://github.com/gpoore/pythontex), and my slightly modified fork of [texfigure](https://github.com/Goobley/texfigure).

### Building the Document

The data needed to build it can be found on the releases page (and should be extracted as a directory named `Data` in this directory).
The necessary Python environment can be installed from `environment.yml` (for conda this would be `conda env create -f environment.yml`).
The document itself is built through `build.sh main.tex`, but will require a path to conda to be provided in the environment variables, for me this is `CMO_CONDA_PATH`, so this short script may require a little modification for your environment.

### Notes

- Due to the amount of radiative transfer reprocessing that needs to happen, a clean build should take ~5 minutes.
- Most threading arguments are hardcoded to 16 for my system.
This is unlikely to be an issue, but something to be aware of if you're from the future with vastly more cores available.
- A lot of the plotting code in here was written for a deadline and doesn't necessarily follow the best practices.
- The `lightweaver` version used for the final submitted build was `v0.7.5`.
- Thanks to @wtbarnes for his [dissertation](https://github.com/wtbarnes/dissertation/) repo, which got me on the right track with pythontex!