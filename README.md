# SotoDehmeltBaumann2026iScience_code
Data analysis and model simulation codes for Soto et al. 2026

## Motor noise estimation

- Run main.py for the entire analysis pipeline. Optionally, running zfish_check_file_names.py generates a metadata summary file for the zebra fish dataset.
- Note the repository only contains the code, but not the underlying dataset, for the analysis.

### Codebase structure

* **config.py**: All configuration-related parameters can be found in this script.
* **function_scripts.py**: All functions used in the codebase are in this script.
* **main.py**: Runs the data analysis pipeline
* **make_figures_pretty.py**: Adjusts the default figure parameters
* **noise_estimation.py**: Analysis script for the motor noise estimations
* **plot_figures.py**: Generates figures
* **preprocess_saccade_data.py**: Preprocesses the raw data to be analyzed further down the line.
* **requirements.txt**: List of libraries with their versions needed for the codebase
* **saccade_overshoots.py**: Analysis script for the saccade overshoots
* **zfish_check_file_names.py**: Generates a metadata summary file for the given dataset. Note this needs to be run separately.

### contact: itunc@uni-koeln.de, ibrahimalperentunc@protonmail.com
