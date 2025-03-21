---

# Power Spectrum and Bispectrum Analysis

This repository contains code for performing a likelihood analysis on power spectrum (`Pk`) and bispectrum (`Bk`) data using the emulator BIKER. The code is designed to work with `yml` configuration files, allowing for flexible and reproducible analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Configuration File](#configuration-file)
5. [Running the Code](#running-the-code)
6. [Output](#output)
7. [Multiprocessing](#multiprocessing)
8. [Testing in Jupyter Notebook](#testing-in-jupyter-notebook)
9. [License](#license)

---

## Overview

The code performs the following tasks:
1. Loads data and covariance matrices from specified paths.
2. Filters the data based on specified multipoles and wavenumber ranges.
3. Computes model predictions using the BIKER emulator.
4. Performs a likelihood analysis using the nested Monte Carlo sampling method implemented in the `nautilus` library.
5. Supports multiprocessing for faster computation.

---

## Requirements ----> UNDER DEVELOPMENT

- Python 3.7 or higher
- Required Python packages:
  - `numpy`
  - `scipy`
  - `yaml`
  - `argparse`
  - `nautilus` (version 1.0.5)
  - `tensorflow` (version 2.15.0 or higher)
  - `keras` (version 2.15.0 or higher)
  - `multiprocessing`

You can install the required packages using `pip`:

```bash
pip install numpy scipy pyyaml argparse nautilus-sampler
```

---

## Installation ----> UNDER DEVELOPMENT

1. Clone the repository:

   ```bash
   git clone https://github.com/cmguandalin/TripoSH-Fitting-v2.git
   cd TripoSH-Fitting-v2
   ```

2. Install the required Python packages (see [Requirements](#requirements)).

---

## Configuration File

The code uses a YAML configuration file to specify input parameters. Below is an example configuration file:

```yaml
data_path: '/path/to/data/'
cov_path: '/path/to/covariance/'   # Optional
cov_file: 'covariance_file.npy'    # Required if cov_path is not provided
data_files:
  0: 'pk_ell0'
  2: 'pk_ell2'
  000: 'bk_ell000'
  202: 'bk_ell202'
rescale: false
number_of_mocks: 1000
k_edges:
  0: [kmin, kmax]
  2: [kmin, kmax]
  000: [kmin, kmax]
  202: [kmin, kmax]
prior:
  param1:
    type: 'Uniform'
    lim: [lower_bound, upper_bound]
  param2:
    type: 'Gaussian'
    lim: [mean, std]
mean_density: 1.0e-3 # This has to be a float, otherwise it is interpreted as a string.
redshift: 0.5
cache_path: '/path/to/cache/files/z0.5/'
```

### Key Configuration Parameters

- `data_path`: Path to the directory containing the data files.
- `cov_path`: Path to the directory containing the covariance file (optional). If not provided, the code assumes the covariance file is in `data_path`.
- `cov_file`: Name of the covariance file (required if there is no `cov_path` variable).
- `data_files`: Dictionary mapping multipoles to their corresponding data files.
- `rescale`: Rescaling factor for the covariance matrix.
- `number_of_mocks`: Number of mocks used for the Hartlap correction.
- `k_edges`: Dictionary specifying the minimum and maximum wavenumbers for each multipole.
- `prior`: Dictionary specifying the priors for the parameters.
- `mean_density`: Mean density of the sample.
- `redshift`: Effectiver redshift of the data.
- `cache_path`: Path to the cache directory for the emulator.
- `path_to_save`: Path to save the results.
- `file_name`: Name of resulting files.

---

## Running the Code

To run the code, use the following command:

```bash
nohup python -u src/inference.py -config config/config.yml > nohup.out 2>&1 &
```
or 
```bash
sbatch scripts/run_fit.sh
```

Log files can be tracked via 
```bash
tail -f nohup.out
```
or
```bash
tail -f logs/jobID_jobName.out
```

### Command-Line Arguments

- `-config`: Path to the configuration file (required).
- `-ncpus`: Number of cpus for multiprocessing runs (optional).
- `-nlive`: Number of live points for sampling (optional).
- `-flive`: Estimate of the fraction of the evidence in the live set (optional).

---

## Output

The code outputs the following:
1. **Sampling Results**: A `.npy` dictionary containing the samples, log_weights, and log_likelihood produced by the `nautilus` sampler, and the priors containing the parameters varied in the analysis.
3. **Log File**: Progress and timing information are constantly printed.

---

## Multiprocessing

The code supports multiprocessing to speed up the likelihood evaluation. By default, it uses all available CPU cores. You can control the number of CPUs by modifying the `ncpus` variable in the code. To run in a single CPU, set `ncpus = 1`. 

---

## Jupyter Notebooks

There are jupyter notebooks available to analyse the chains

---

## License

This project is licensed under the GNU General Public License. See the [LICENCE](LICENCE) file for details.

---

## Acknowledgments

- The `nautilus` library for MCMC sampling.
- The emulator used for computing power spectrum and bispectrum predictions.
