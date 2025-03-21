from nautilus import Prior
from scipy.stats import norm, uniform
import numpy as np

class Likelihood:
    def __init__(self, priors_dict, model_function):
        """
        Initialise the Likelihood class.

        Args:
            priors_dict (dict): Dictionary of priors for the parameters.
            compute_model_function (callable): Function to compute the model predictions.
        """
        self.priors_dict = priors_dict
        self.model_function = model_function

    def initialise_prior(self):
        """
        Initialise the prior distributions based on the priors dictionary.

        Returns:
            Prior: A Prior object from the nautilus library.
        """
        prior = Prior()  # Initialise nautilus Prior object

        for param, prior_info in self.priors_dict.items():

            if prior_info['type'] == 'Fix':
                # Skip fixed parameters
                continue

            if prior_info['type'] in ['Uni', 'Uniform']:
	        # Uniform distribution
                lower, upper = prior_info['lim']
                prior.add_parameter(param, dist=(lower, upper))
            elif prior_info['type'] in ['Gauss', 'Gaussian']:
                # Gaussian distribution
                mean, std = prior_info['lim'][0], prior_info['lim'][1]
                prior.add_parameter(param, dist=norm(loc=mean, scale=std))
            else:
                raise ValueError(f"Unknown prior type: {prior_info['type']}")

        return prior

    #def ln_prob(self, param_dict, data_, icov_):
    def ln_prob(self, theta, data_, icov_):
        """
        Compute the log-probability for the given parameters.

        Args:
            param_dict (dict): Dictionary of parameter names and values.
            data_ (np.ndarray): Observed data vector.
            icov_ (np.ndarray): Inverse covariance matrix.

        Returns:
            float: Log-probability.
        """
        # Convert the parameter dictionary to a numpy array
        #theta = np.array([param_dict[param] for param in sorted(param_dict.keys())])
        m = self.model_function(theta)
        diff = m - data_
        chi2_try = np.dot(diff.T, np.dot(icov_, diff))

        return -0.5 * chi2_try
