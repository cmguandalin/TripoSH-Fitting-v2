import argparse
import os,sys
import yaml
import numpy as np
from time import time
import pocomc as pc
import data_loader as dload
import covariance_loader as cload
import likelihood as clike
import model
from datetime import datetime
from multiprocessing import Pool

os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
sys.stderr = sys.stdout

# Global variables (avoid passing them as arguments) - this should make the sampler faster in slurm
global global_full_data, global_inv_cov, global_likelihood
global_full_data = None
global_inv_cov = None
global_likelihood = None

def likelihood_wrapper(theta):
    return global_likelihood.ln_prob(theta, global_full_data, global_inv_cov)

if __name__ == '__main__':

    time_i = time()

    ##############################
    # LOADING CONFIGURATION FILE #
    ##############################
    parser = argparse.ArgumentParser(description='Configuration file to load')
    parser.add_argument('-config', '-c', '-C', type=str, help='config file', required=True,dest='config')
    parser.add_argument('-ncpus', type=int, help='Number of CPUs in a PC to use.', required=False, default=1)
    cmdline = parser.parse_args()

    print(f'Using {cmdline.config}')

    with open(cmdline.config, 'r') as file:
        config = yaml.safe_load(file)

    # Get the path for data and covariance files
    data_path = config['data_path']
    # Get the files names
    data_files = config['data_files']
    # Get the covariance path
    cov_path = config.get('cov_path')
    if not cov_path:
        # If only the file name is given, it's assumed the data path is the same as the covariance path
        cov_file = config['cov_file']
        cov_path = data_path+cov_file
    rescale = config['rescale']
    # Number of mocks for hartlap correction
    number_of_mocks = config['number_of_mocks']
    # Rescaling factor for the covariance
    rescale = config['rescale']
    # Minimum and maximum wavenumbers to consider
    k_edges = config['k_edges']
    # The priors
    priors = config['prior']
    path_to_save = config['path_to_save']
    file_name = config['file_name']

    # For the emulator
    multipoles   = list(data_files.keys())
    mean_density = config['mean_density']
    redshift     = config['redshift']
    cache_path   = config['cache_path']

    #For window matrices
    wcmat_path = config['window_path']
    wcmat_files_pk = os.path.join(wcmat_path, config['window_files']['pk'])
    wcmat_files_000 = os.path.join(wcmat_path, config['window_files']['000'])

    #######################
    # CLEANING PARAMETERS #
    #######################

    parameters_to_be_varied = priors.copy()
    for param, prior_info in list(priors.items()):
        if prior_info['type'] == 'Fix':
            del parameters_to_be_varied[param]

    #############
    # LOAD DATA #
    #############
    loader = dload.DataLoader(data_path,data_files,multipoles)
    loader.load_data(k_edges)
    data = loader.get_data()
    full_k, full_data = loader.get_concatenated_data()

    ###################
    # LOAD COVARIANCE #
    ###################
    cov_loader = cload.CovarianceLoader(cov_path, multipoles, k_edges, rescale)
    cov_loader.process()
    covariance = cov_loader.get_covariance()

    # Apply Hartlap correction factor and invert covariance
    hartlap = (number_of_mocks - len(full_data) - 2) / (number_of_mocks - 1)
    inv_cov = hartlap * np.linalg.inv(covariance)

    ################
    # LOAD WINDOW MATRICES #
    ################
    window_loader = dload.WindowLoader(wcmat_files_000, wcmat_files_pk, multipoles)
    wcmat_dict = window_loader.load_windows()

    print(wcmat_dict)

    ################
    # MODEL VECTOR #
    ################
    # Initialise the emulator
    calculator = model.PkBkCalculator(multipoles, mean_density, redshift, cache_path, fixed_params=['omega_cdm', 'omega_b', 'ln10^{10}A_s', 'n_s'], rescale_kernels=True, ordering=1)
    model_function = model.ModellingFunction(priors, data, calculator, multipoles, wcmat_dict)

    ##############
    # LIKELIHOOD #
    ##############
    likelihood = clike.Likelihood(priors, model_function.compute_model_vector)
    prior = likelihood.initialise_prior()

    # Assign to global variables
    global_full_data = full_data
    global_inv_cov = inv_cov
    global_likelihood = likelihood

    ##################
    # START SAMPLING #
    ##################

    # number of effective particles
    neff = 4000
    # number of effectively independent samples
    ntot = 20000

    if cmdline.ncpus is not None:
        ncpus = int(cmdline.ncpus)
    else:
        ncpus = 1

    print(f'Starting sampling at {datetime.now()} with {ncpus} CPUs. \n')

    if ncpus > 1:
        with Pool(ncpus) as pool:
            sampler = pc.Sampler(
                prior=prior,
                likelihood=likelihood_wrapper,
                n_effective=neff,
                pool=pool,
                output_dir=path_to_save,
                output_label=file_name
            )
            sampler.run(n_total=ntot, progress=True, save_every=200)

    else:
        sampler = pc.Sampler(
            prior=prior,
            likelihood=likelihood_wrapper,
            n_effective=neff,
            output_dir=path_to_save,
            output_label=file_name
        )
        sampler.run(n_total=ntot, progress=True, save_every=200)

    samples, weights, logl, logp = sampler.posterior()

    print(f"Sampling ended at: {datetime.now()}")

    # Save results
    os.makedirs(path_to_save, exist_ok=True)

    print(f"Results saved to {os.path.join(path_to_save, file_name + '.npy')}")

    results = {}
    results['priors'] = parameters_to_be_varied
    results['samples'] = samples
    results['weights'] = weights
    results['logl'] = logl
    results['logp'] = logp

    np.save(os.path.join(path_to_save, file_name + '.npy'), results)

    time_f = time()

    print('Sampling efficiency:', sampler.results["efficiency"])
    print('Time to estimate (in minutes):', np.round((time_f-time_i)/60,2))
