import argparse
import os
import yaml
import numpy as np
from time import time
from multiprocessing import Pool, cpu_count
import pocomc as pc
import data_loader as dload
import covariance_loader as cload
import likelihood as clike
import model

config_file = '/Users/austerlitz/TripoSH-Factory/my_project/debug/config/lrg_test.yml'

# Define the log-probability function for the sampler
def likelihood_wrapper(theta, data_, icov_, instance):
    return instance.ln_prob(theta, data_, icov_)

if __name__ == '__main__':
    
    time_i = time()

    #####################################
    # LOADING CONFIGURATION FOR FITTING #
    #####################################
    
    #parser = argparse.ArgumentParser(description='Configuration file to load')
    #parser.add_argument('-config', type=str, help='config file', required=True)
    #cmdline = parser.parse_args()

    #print(f'Using {cmdline.config}')
    
    with open(config_file, 'r') as file:
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
    
    # For the emulator
    multipoles   = list(data_files.keys())
    mean_density = config['mean_density']
    redshift     = config['redshift']
    cache_path   = config['cache_path']    


    #######################
    # CLEANING PARAMETERS #
    #######################

    # Iterate over a copy of the dictionary to avoid modifying it while iterating
    for param, prior_info in list(priors.items()):
        if prior_info['type'] == 'Fix':
            del priors[param]

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
    calculator = model.PkBkCalculator(multipoles, mean_density, redshift, cache_path)
    # NEED TO ADD A CLASS TO THE MODEL.PY SUCH THAT CALCULATOR CAN BE REPLACED 
    # BY CLASS-PT, OR SOMETHING ELSE -- E.G. VELOCILEPTORS
    # Initialise the model function
    # (it is basically a wrapper for the model function used in the likelihood)
    # The calculator gives the Pk and Bk from the emulator, while the variable
    # below fetches the concatenated model, with the k modes coming from the data
    get_model = model.ModellingFunction(priors, data, calculator, multipoles)
    # Use in the likelihood: test_model = get_model.compute_model_vector(theta)

    ##############
    # LIKELIHOOD #
    ##############
    likelihood = likeli.Likelihood(priors, get_model.compute_model_vector)
    # Initialise the prior
    prior = likelihood.initialise_prior()
    
    ##################
    # START SAMPLING #
    ##################
    # Set up the sampler
    ncpus = int(os.getenv('SLURM_CPUS_PER_TASK', cpu_count()))
    with Pool(ncpus) as pool:
        # Define the log-probability function for the sampler
        #def ln_prob(theta, data_, icov_):
        #    likelihood = likeli.Likelihood(priors, get_model.compute_model_vector)
        #    return likelihood.ln_prob(theta, data_, icov_)
            
        sampler = pc.Sampler(
            prior=prior,
            #likelihood=ln_prob,
            likelihood=likelihood_wrapper,
            likelihood_args=[full_data, inv_cov, likelihood],
            pool=pool,
            n_effective=10000
        )
    
        # Run the sampler
        sampler.run(progress=True)

    samples, logl, logp = sampler.posterior(resample=True)
    
    #print('Saving chains at')
    
    time_f = time()

    print('Time to estimate:', (time_f-time_i)/60)