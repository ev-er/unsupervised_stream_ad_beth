# unsupervised_stream_anomaly_detection_beth

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Performance evaluation of unsupervised stream anomaly detection algorithms on BETH dataset

## Project Organization

```
├── LICENSE            <- License
├── Makefile           <- Makefile with convenience commands like `make data` or `make features`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. 
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         usad_beth and configuration for tools like black
|
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── usad_beth   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes usad_beth a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling.py             <- Code to run experiments          
    │
    └── plots.py                <- Code to create visualizations
```


See notebooks for more information.

all_in_one_notebook.ipynb - can be copied for using standalone

experiments.ipynb - short notebook with more imports from this project

--------

