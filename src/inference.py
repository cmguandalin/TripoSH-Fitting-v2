
import argparse
import os, sys
import yaml
import numpy as np

import data_loader as dload
import covariance_loader as cload
import likelihood as clike
import model

import h5py
from datetime import datetime
from time import time

from nautilus import Prior
from nautilus import Sampler

'''
    ~*~*~*~ history ~*~*~*~

v2) a) added fixed parameters to likelihood;
    b) saving parameters_to_be_varied instead of the priors to the .npy result.

'''

''' STARTING '''

# NEED THIS OTHERWISE SAMPLING DOES NOT RUN!
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

sys.stderr = sys.stdout

# Define the log-probability function for the sampler
def likelihood_wrapper(theta, data_, icov_, instance):
    return instance.ln_prob(theta, data_, icov_)

if __name__ == '__main__':

    time_i = time()

    ##############################
    # LOADING CONFIGURATION FILE #
    ##############################

    parser = argparse.ArgumentParser(description='Configuration file to load')
    parser.add_argument('-config', '-c', '-C', type=str, help='config file', required=True,dest='config')
    parser.add_argument('-ncpus', type=int, help='Number of CPUs in a PC to use.', required=False)
    parser.add_argument('-nlive', type=int, help='Number of live points for sampling.', required=False)
    parser.add_argument('-flive', type=int, help='Estimate of the fraction of the evidence in the live set.', required=False)
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


    #######################
    # CLEANING PARAMETERS #
    #######################

    # Iterate over a copy of the dictionary to avoid modifying it while iterating
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
    # MODEL VECTOR #
    ################
    # Initialise the emulator
    calculator = model.PkBkCalculator(multipoles, mean_density, redshift, cache_path, fixed_params=['n_s'], rescale_kernels=True, ordering=1)
    model_function = model.ModellingFunction(priors, data, calculator, multipoles)

    ########################
    # LIKELIHOOD AND PRIOR #
    ########################
    likelihood = clike.Likelihood(priors, model_function.compute_model_vector)

    def likelihood_nautilus(theta):
        return likelihood_wrapper(theta, full_data, inv_cov, likelihood)

    prior = likelihood.initialise_prior()

    ##################
    # START SAMPLING #
    ##################

    checkpoint_file = os.path.join(path_to_save, file_name + '_checkpoint.h5')

    if os.path.exists(checkpoint_file):
        print(f"Warning: Checkpoint {checkpoint_file} already exists. Resuming from existing chain.")
        resume_checkpoint = True
    else:
        resume_checkpoint = False
        print(f"Creating a new checkpoint file: {checkpoint_file}")

    if cmdline.ncpus:
        ncpus = int(cmdline.ncpus)
    else:
        ncpus = 1

    if cmdline.nlive:
        nlive = int(cmdline.nlive)
    else:
        nlive = 1000

    if cmdline.flive:
        flive = int(cmdline.flive)
    else:
        flive = 0.1

    print(f'Starting sampling at {datetime.now()} with {ncpus} CPUs, {nlive} live points and f_live={flive}. \n')

    if ncpus > 1:
        sampler = Sampler(
            prior=prior,
            likelihood=likelihood_nautilus,
            pass_dict=False,
            n_live=nlive,
            pool=ncpus,
            filepath=checkpoint_file
        )

        sampler.run(f_live=flive,verbose=True,discard_exploration=True)
        samples, log_weights, log_like = sampler.posterior()
    else:
        sampler = Sampler(
            prior=prior,
            likelihood=likelihood_nautilus,
            pass_dict=False,
            n_live=nlive,
            filepath=checkpoint_file
        )

        sampler.run(f_live=flive,verbose=True,discard_exploration=True)
        samples, log_weights, log_like = sampler.posterior()

    print(f"Sampling ended at: {datetime.now()}")

    # Save the final results in .npy format
    results = {
        'samples': samples,
        'log_w': log_weights,
        'log_l': log_like,
        'priors': parameters_to_be_varied
    }

    np.save(os.path.join(path_to_save, file_name + '_results.npy'), results)

    print(f"Results saved to {os.path.join(path_to_save, file_name + '_results.npy')}")

    time_f = time()

    print('Time to estimate (in minutes):', np.round((time_f-time_i)/60,2))
