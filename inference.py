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

os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# Define the log-probability function for the sampler
def likelihood_wrapper(theta, data_, icov_, instance):
    return instance.ln_prob(theta, data_, icov_)

if __name__ == '__main__':
    
    time_i = time()

    ##############################
    # LOADING CONFIGURATION FILE #
    ##############################
    
    parser = argparse.ArgumentParser(description='Configuration file to load')
    parser.add_argument('-config', type=str, help='config file', required=True)
    cmdline = parser.parse_args()

    print(f'Using {cmdline.config}')
    
    with open(cmdline.config, 'r') as file:
        config = yaml.safe_load(file)

    #########################
    # DEFINE MAIN VARIABLES #
    #########################

    # 1) Path for data files
    data_path = config['data_path']
    # 2) Files names
    data_files = config['data_files']
    # 3) Covariance path
    cov_path = config.get('cov_path')
    if not cov_path:
        # If only the file name is given, it is assumed the data path is the same as the covariance path
        cov_file = config['cov_file']
        cov_path = data_path+cov_file
    # 4) Rescaling factor for the covariance
    rescale = config['rescale']
    # 5) Number of mocks for hartlap correction
    number_of_mocks = config['number_of_mocks']
    # 6) Minimum and maximum wavenumbers to consider
    k_edges = config['k_edges']
    # 7) The priors
    priors = config['prior']
    # 8) Path to save the results
    path_to_save = config['path_to_save']
    # 9) File name for results
    file_name = config['file_name']
    
    # For the emulator:
    # 1) Multipoles we are working with
    multipoles   = list(data_files.keys())
    # 2) Mean density for shot noise
    mean_density = config['mean_density']
    # 3) Effective redshift of the sample
    redshift     = config['redshift']
    # 4) Path to emulator files
    cache_path   = config['cache_path']    

    #######################
    # CLEANING PARAMETERS #
    #######################

    # Removing parameters not used in the analysis
    for param, prior_info in list(priors.items()):
        if prior_info['type'] == 'Fix':
            del priors[param]

    
    # ~*~*~ PREPARING FOR SAMPLING ~*~*~ #

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
    # Note: The calculator will give the P and B from the emulator, while the
    # model_function gives the concatenated model, with the k modes coming from the data
    
    # Initialise the emulator
    calculator = model.PkBkCalculator(multipoles, mean_density, redshift, cache_path, fixed_params=None)
    # Initialise the model function
    model_function = model.ModellingFunction(priors, data, calculator, multipoles)

    ##############
    # LIKELIHOOD #
    ##############
    # Initialise the likelihood
    likelihood = clike.Likelihood(priors, model_function.compute_model_vector)
    # Initialise the prior
    prior = likelihood.initialise_prior()

    # ~*~*~ START SAMPLING ~*~*~ #
    
    ###############
    # SET SAMPLER #
    ###############
    
    ncpus = int(os.getenv('SLURM_CPUS_PER_TASK', cpu_count()))

    print(f'Starting sampling with {ncpus} CPUs')
          
    with Pool(ncpus) as pool:
            
        sampler = pc.Sampler(
            prior=prior,
            likelihood=likelihood_wrapper,
            likelihood_args=[full_data, inv_cov, likelihood],
            pool=pool,
            n_effective=10000,
            output_dir=path_to_save,
            output_label=file_name
        )
    
        # Run the sampler
        sampler.run(progress=True,save_every=50)

    samples, logl, logp = sampler.posterior(resample=True)

    ##########
    # SAVING #
    ##########
    
    os.makedirs(path_to_save, exist_ok=True)
    
    print(f'Saving {file_name} at {path_to_save}')
    
    output = {}
    output['priors']  = priors
    output['samples'] = samples
    output['logl']    = logl
    output['logp']    = logp
    
    np.save(path_to_save+file_name+'.npy',output)

    time_f = time()

    print('Time to estimate (in minutes):', np.round((time_f-time_i)/60,2))