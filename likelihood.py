from scipy.stats import norm, uniform
import pocomc as pc
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
            pc.Prior: A Prior object from the pocomc library.
        """
        prior_list = []
        for param, prior_info in self.priors_dict.items():
            if prior_info['type'] in ['Uni', 'Uniform']:
                # Uniform distribution
                lower, upper = prior_info['lim']
                prior_list.append(uniform(lower, upper - lower))
            elif prior_info['type'] in ['Gauss', 'Gaussian']:
                # Gaussian distribution
                mean, std = prior_info['lim'][0], prior_info['lim'][1]
                prior_list.append(norm(mean, std))
            else:
                raise ValueError(f"Unknown prior type: {prior_info['type']}")
        
        return pc.Prior(prior_list)

    def ln_prob(self, theta, data_, icov_):
        """
        Compute the log-probability for the given parameters.

        Args:
            theta (np.ndarray): Array of parameter values.
            data_ (np.ndarray): Observed data vector.
            icov_ (np.ndarray): Inverse covariance matrix.

        Returns:
            float: Log-probability.
        """
        m = self.model_function(theta)
        diff = m - data_
        chi2_try = np.dot(diff.T, np.dot(icov_, diff))
        
        return -0.5 * chi2_try