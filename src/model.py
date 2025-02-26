""" 
THEORETICAL MODELLING
=====================

Example of usage: 

    import numpy as np
    import model
    
    multipoles   = ['0','2','000','202'] <---- all multipoles you want to compute
    mean_density = 1e-3
    redshift     = 0.8
    cache_path   = '/path/to/TripoSH-Factory/TripoSH-Fitting/example/z0.8/' <---- path to where emulators are stored

    calculator   = model.PkBkCalculator(multipoles, mean_density, redshift, cache_path)

    params = {...} <---- a dictionary with the parameters you want
    Pk_0   = calculator.pk_from_emulator(params, '0')
    Pk_2   = calculator.pk_from_emulator(params, '2')
    Bk_000 = calculator.pk_from_emulator(params, '000')
    Bk_202 = calculator.pk_from_emulator(params, '202')
    
Then plot these functions within your desired range of wavemodes.

Use calculator.help() for a quick documentation.

""" 

import numpy as np
from scipy.interpolate import interp1d
import bicker.emulator as BICKER
from time import time 
import os
import re

group = [
            ['c2_b2_f', 'c2_b1_b2',  'c2_b1_b1',  'c2_b1_f', 'c2_b1_f', 'c1_b1_b1_f', 
                'c1_b2_f', 'c1_b1_b2', 'c1_b1_b1', 'c2_b1_b1_f', 'c1_b1_f'],
            ['c2_b1_f_f', 'c1_f_f', 'c1_f_f_f', 'c2_f_f', 'c2_f_f_f', 'c1_b1_f_f'],
            ['c1_c1_f_f', 'c2_c2_b1_f',  'c2_c1_b1_f',  'c2_c1_b1', 'c2_c1_b2', 
                'c2_c2_f_f', 'c1_c1_f', 'c2_c2_b1', 'c2_c2_b2', 'c2_c2_f', 'c2_c1_b1_f', 
                'c2_c1_f', 'c1_c1_b1_f', 'c1_c1_b1', 'c1_c1_b2', 'c1_c1_f_f'],
            ['c1_c1_bG2', 'c2_c2_bG2', 'c2_c1_bG2'],
            ['c1_b1_bG2', 'c1_bG2_f', 'c2_bG2_f', 'c2_b1_bG2'],
            ['b1_f_f', 'b1_b1_f_f', 'b1_b1_b2', 'b2_f_f', 'b1_b1_b1', 'b1_b1_b1_f', 
                'b1_b1_f', 'b1_f_f_f', 'f_f_f', 'f_f_f_f', 'b1_b2_f'],
            ['bG2_f_f', 'b1_b1_bG2', 'b1_bG2_f']
        ]

group_shot = [
                'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2',
                'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2',
            ]

kernel_name_Bk = [
                    'b1_b1_b1', 'b1_b1_b2','b1_b1_bG2','b1_b1_f','b1_b1_b1_f','b1_b1_f_f',
                    'b1_b2_f','b1_bG2_f','b1_f_f',
                    'b1_f_f_f','b2_f_f','bG2_f_f','f_f_f','f_f_f_f',
                    'c1_b1_b1','c1_b1_b2','c1_b1_bG2','c1_b1_f','c1_b1_b1_f','c1_b1_f_f','c1_b2_f',
                    'c1_bG2_f','c1_f_f','c1_f_f_f','c1_c1_b1','c1_c1_b2','c1_c1_bG2','c1_c1_f',
                    'c1_c1_b1_f','c1_c1_f_f','c2_b1_b1','c2_b1_b2','c2_b1_bG2','c2_b1_f','c2_b1_b1_f',
                    'c2_b1_f_f','c2_b2_f','c2_bG2_f','c2_f_f','c2_f_f_f','c2_c1_b1','c2_c1_b2',
                    'c2_c1_bG2','c2_c1_f','c2_c1_b1_f','c2_c1_f_f','c2_c2_b1','c2_c2_b2','c2_c2_bG2',
                    'c2_c2_f','c2_c2_b1_f','c2_c2_f_f',
                    'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2', 
                    'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2',
                    'fnlloc_b1_b1_b1','fnlloc_b1_b1_f','fnlloc_b1_f_f','fnlloc_f_f_f',
                    'fnlequi_b1_b1_b1','fnlequi_b1_b1_f','fnlequi_b1_f_f','fnlequi_f_f_f',
                    'fnlortho_b1_b1_b1','fnlortho_b1_b1_f','fnlortho_b1_f_f','fnlortho_f_f_f',
                    'fnlortho_LSS_b1_b1_b1','fnlortho_LSS_b1_b1_f','fnlortho_LSS_b1_f_f','fnlortho_LSS_f_f_f',]

params_sorted = ['omega_cdm','omega_b','h','ln10^{10}A_s','n_s','f','b1','b2','bG2','bGamma3', 'c0', 'c2pp', 'c4pp', 'c1', 'c2','ch','Pshot','a0','Bshot','fnlloc','fnlequi','fnlortho']

class PkBkCalculator:
    """
    Class to compute power spectrum and bispectrum models given EFT parameters.

    Attributes:
    -----------
    * cache_path : str
        Path to the cache where the emulator data is stored.
    * mean_density : float
        The mean density of the universe.
    * zcen : float
        The central redshift value.
    * fixed_params : list of strings
        Contains the parameters that were not considered in the training.
    * rescale_kernels : bool
        If the training was done with kernels scaled by As^n, then they should be rescaled back.
    * multipoles_pk : list
        List of multipoles for the power spectrum.
    * multipoles_bk : list
        List of multipoles for the bispectrum.
    * kemul_pk : array
        The k values for the power spectrum emulator.
    * kemul_bk : array
        The k values for the bispectrum emulator.
    * emulator_pk : dict
        Emulators for the power spectrum.
    * emulator_bk : dict
        Emulators for the bispectrum.
    
    Main functions:
    ---------------
    * __init__(multipoles, mean_density, redshift, cache_path, fixed_params, rescaled)
        Initialises the PkBkCalculator with the given parameters.
    
    * kernels_from_emulator(pars, ell)
        Fetches kernels from the bispectrum emulator for a given ell.
    
    * pk_from_emulator(pars, ell)
        Computes the power spectrum model for a given multipole.
    
    * bk_from_emulator(pars, l1l2L)
        Computes the bispectrum model for a given set of multipoles.
    
    * help()
        Provides documentation and usage instructions for the class.
    """
    
    def __init__(self, multipoles, mean_density, redshift, cache_path, fixed_params=['n_s'], rescale_kernels=True, ordering=0):
        """
        Parameters:
        - multipoles (list)
        - mean_density (float)
        - redshift (float)
        - cache_path (str)
        - fixed_params (list of str)
        - rescale_kernels (bool)
        """

        print('Initialising PkBkCalculator.\n')
        time_i = time()

        self.cache_path = cache_path
        self.mean_density = mean_density
        self.zcen = redshift
        self.fixed_params = fixed_params
        self.rescale_kernels = rescale_kernels
        self.ordering = ordering

        #############################################################
        # Check if the redshift is the one used to train the emulator
        self._check_redshift()

        #############################
        # Set the multipole variables
        self.multipoles_pk = []
        self.multipoles_bk = []
        for i in multipoles:
            if len(i)==1:
                self.multipoles_pk.append(i)
            elif len(i)==3:
                self.multipoles_bk.append(i)
            else:
                print('Unrecognised multipole detected.')
        
        if self.multipoles_pk:
            self.kemul_pk = np.loadtxt(os.path.join(self.cache_path, 'powerspec/k_emul.txt'))
        if self.multipoles_bk:
            self.kemul_bk = np.loadtxt(os.path.join(self.cache_path, 'bispec/k_emul.txt'))

        #self._initialise_multipoles(multipoles) <----- this function is not working due to self.kemul

        ######################
        # Initialise emulators
        self._initialise_emulators()
        
        time_f = time()

        ######
        # Done
        print(f'Total time to initialise the calculator: {round(time_f-time_i,2)} seconds.') 
        print(f'You can now compute the {self.multipoles_pk} power spectrum and {self.multipoles_bk} bispectrum multipoles.')
        print(f'Use the function help() for further guidance.')
        
    '''
        Helper functions
    '''
    
    def _check_redshift(self):
        # Check if provided redshift is the same as the one used to train the emulator. 
        pattern = r'z(\d+\.\d+)'
        emulator_redshift = re.search(pattern, self.cache_path).group(1)
        if self.zcen != float(emulator_redshift):
            raise ValueError(f'Redshift {self.zcen} does not match the one used by the emulator: {emulator_redshift}.')
    

    def _initialise_emulators(self):
        # Initialise emulators for power spectrum and/or bispectrum
        if self.multipoles_pk:
            self.emulator_pk = {ell: BICKER.power(ell, self.kemul_pk, self.cache_path) for ell in self.multipoles_pk}
        
        if self.multipoles_bk:
            self._initialise_bk_emulators()
     
    def _initialise_bk_emulators(self):
        # Initialize bispectrum emulators from cache_path
        self.group_to_emul = self._get_groups_to_emulate()
        self.emulator_bk = {}

        for ell in self.multipoles_bk: 
            self.emulator_bk[ell] = {}
            for gp in self.group_to_emul:
                emulator_type = 'shot' if gp == 8 else gp
                self.emulator_bk[ell][gp] = BICKER.component_emulator(emulator_type, ell, self.kemul_bk, self.cache_path)
        
    def _get_groups_to_emulate(self):
        # Determine which groups to emulate based on kernel names
        group_to_emul = []
        for gp, kernels in enumerate(group):
            if any(kernel in kernel_name_Bk for kernel in kernels):
                group_to_emul.append(gp)
        if 'Bshot' in params_sorted:
            group_to_emul.append(8)
        return group_to_emul
    
    def _get_cosmo_params(self, pars):
        # Helper function to get cosmological parameters from the full EFT parameters
        # Used in the pk_from_emulator function.
        default_ns = 0.9649

        if self.fixed_params is None: 
            # n_s was included in the training of the emulator
            if 'n_s' not in pars:
                # but is not varied in the MCMC analysis

                if self.ordering==0:
                    return [pars['omega_cdm'], pars['omega_b'], pars['h'], pars['ln10^{10}A_s'], default_ns]
                else:
                    return [pars['omega_b'], pars['h'], pars['omega_cdm'], pars['ln10^{10}A_s'], default_ns]
            else:
                # and it is varied in the MCMC analysis
                if self.ordering==0:
                    return [pars['omega_cdm'], pars['omega_b'], pars['h'], pars['ln10^{10}A_s'], pars['n_s']]
                else:
                    return [pars['omega_b'], pars['h'], pars['omega_cdm'], pars['ln10^{10}A_s'], pars['n_s']]
        elif 'n_s' in self.fixed_params:
            # n_s was not included in the training of the emulator, therefore it cannot be varied
            if 'n_s' not in pars:
                if self.ordering==0:
                    return [pars['omega_cdm'], pars['omega_b'], pars['h'], pars['ln10^{10}A_s']]
                else:
                    return [pars['omega_b'], pars['h'], pars['omega_cdm'], pars['ln10^{10}A_s']]
            else:
                raise ValueError(f"n_s was not included in the training of the emulator, therefore it cannot be varied. Fix this parameter to its fiducial value in the sampling procedure.")
                
    '''
        Main functions
    '''
    
    def kernels_from_emulator(self, pars, ell): 
        """
        Compute kernels from the emulator for given EFT parameters and multipole ell.

        Parameters:
        - pars (dict): EFT parameters.
        - ell (str): Multipole.

        Returns:
        - kbins (ndarray): Binned k values.
        - kernels (dict): Computed kernels.
        """
        cosmo_pars = self._get_cosmo_params(pars)
        self.kernels = {}

        # For the rescaled kernels
        if self.rescale_kernels:
            As = (np.exp(pars['ln10^{10}A_s'])*1e-10)
            As2 = As**2.0
        else:
            As = 1.0
            As2 = 1.0

        for gp in self.emulator_bk[ell].keys():
            predictions = self.emulator_bk[ell][gp].emu_predict(cosmo_pars, split=True)

            if gp == 8:
                for i, kern in enumerate(group_shot):
                    self.kernels[kern] = np.reshape(predictions[i], predictions[i].shape[1])
                    self.kernels[kern] *= As
            else:
                for i, kern in enumerate(group[gp]):
                    self.kernels[kern] = np.reshape(predictions[i], predictions[i].shape[1])
                    self.kernels[kern] *= As2

        return self.emulator_bk[ell][gp].kbins, self.kernels
            
    def pk_from_emulator(self, pars, ell):
        """
        Get power spectrum from emulator for given EFT parameters and multipole ell.

        Parameters:
        - pars (dict): EFT parameters.
        - ell (str): Multipole.

        Returns:
        - interp_function (callable): Interpolated power spectrum multipole.
        """
        
        if not len(ell) == 1:
            raise ValueError(f'{ell} is not a valid power spectrum multipole.')
        
        # Extract nuisance parameters
        b1, b2, bG2, bGamma3, ch = pars['b1'], pars['b2'], pars['bG2'], pars['bGamma3'], pars['ch']
        Pshot, a0 = pars['Pshot'], pars['a0']
        self.Pstoch = (1 + Pshot + a0 * self.emulator_pk[ell].kbins**2.0) / self.mean_density
        cosmo_pars = self._get_cosmo_params(pars)

        # Determine counterterms for each multipole and compute power spectrum from the emulator
        if ell == '0':
            c0 = pars.get('c0', 0.0)
            self.Pk_ell = self.emulator_pk[ell].emu_predict(cosmo_pars, [b1, b2, bG2, bGamma3, ch, c0])[0] + self.Pstoch
        elif ell == '2':
            c2pp = pars.get('c2pp', 0.0)
            self.Pk_ell = self.emulator_pk[ell].emu_predict(cosmo_pars, [b1, b2, bG2, bGamma3, ch, c2pp])[0]
        elif ell == '4':
            c4pp = pars.get('c4pp', 0.0)
            self.Pk_ell = self.emulator_pk[ell].emu_predict(cosmo_pars, [b1, b2, bG2, bGamma3, ch, c4pp])[0]
        else:
            raise ValueError('Unrecognised multipole.')

        self.interp_function = interp1d(self.kemul_pk, self.Pk_ell, kind='cubic', fill_value='extrapolate')
        
        return self.interp_function
            
    def bk_from_emulator(self, pars,l1l2L):
        """
        Compute the bispectrum from the emulator given the EFT parameters and multipoles l1, l2, L.

        Parameters:
        - pars (dict): Dictionary containing EFT parameters.
        - l1l2L (str): String representing the multipole combination for bispectrum calculation.

        Returns:
        - interp_function (callable): Interpolated bispectrum multipole.
        """
        
        if not len(l1l2L) == 3:
            raise ValueError(f'{l1l2L} is not a valid bispectrum multipole.')
            
        b1 = pars.get('b1',1.0)
        b2 = pars.get('b2',0.0)
        bG2 = pars.get('bG2',0.0)
        c1 = pars.get('c1',0.0)
        c2 = pars.get('c2',0.0)
        Pshot = pars.get('Pshot',0.0)
        Bshot = pars.get('Bshot',0.0)

        # Get kernels for bispectrum emulator
        self.kernels_k, self.kernels_Bk = self.kernels_from_emulator(pars, l1l2L)
        self.bk_model = np.zeros(len(self.kernels_Bk['b1_b1_b1']))

        # Iterate over the kernels to compute the bispectrum model
        for b, values in self.kernels_Bk.items():
            bias = 1
            if 'b1' in b:
                bias *= b1 ** b.count('b1')
            if 'b2' in b:
                bias *= b2
            if 'bG2' in b:
                bias *= bG2
            if 'c1' in b:
                bias *= c1 ** b.count('c1')
            if 'c2' in b:
                bias *= c2 ** b.count('c2')
            if 'Pshot' in b:
                bias *= (1 + Pshot) / self.mean_density
            if 'Bshot' in b:
                bias *= Bshot / self.mean_density
            if 'fnlequi' in b:
                bias *= fnlequi
            if 'fnlortho' in b:
                bias *= fnlortho
            
            # Get the bias-weighted kernel
            self.bk_model += bias * values
        
        if Pshot != 0: 
            self.bk_model += ((1+Pshot)/self.mean_density)**2

        self.interp_function = interp1d(self.kernels_k, self.bk_model, kind='cubic', fill_value='extrapolate')
        
        return self.interp_function

    def help(self):
        """
        Help Function
        =============

        This function provides an overview of the PkBkCalculator class, its attributes, 
        and methods.

        Usage:
        ------
        calculator = PkBkCalculator(multipoles, mean_density, redshift, cache_path, fixed_params, rescale)
        
        Methods:
        --------
        1. kernels_from_emulator(pars, ell)
            - Fetches kernels from the emulator for a given ell.
            - Parameters: 
                pars (dict): A dictionary containing EFT parameters.
                ell (str): The multipole to compute (e.g., '000', '202').
        
        2. pk_from_emulator(pars, ell)
            - Computes the power spectrum model for a given multipole.
            - Parameters:
                pars (dict): A dictionary containing EFT parameters.
                ell (str): The multipole to compute (e.g., '0', '2', '4').

        3. bk_from_emulator(pars, l1l2L)
            - Computes the bispectrum model for a given set of multipoles.
            - Parameters:
                pars (dict): A dictionary containing EFT parameters.
                l1l2L (str): A string representing the bispectrum multipoles.

        Example:
        --------
        ```
        calculator = PkBkCalculator(multipoles=['0', '2','000'], mean_density=1e-3, redshift=0.8, cache_path='path/to/z0.8')
        Pk_0 = calculator.pk_from_emulator(pars, '2')
        Bk_202 = calculator.bk_from_emulator(pars, '202')
        ```
        """
        print(self.help.__doc__)

class ModellingFunction:
    def __init__(self, priors, data, calculator, multipoles):
        """
        Initialize the ModellingFunction class.

        Args:
            priors (dict): Dictionary of priors for the parameters.
            data (dict): Dictionary containing the loaded data.
            calculator: The emulator calculator object (e.g., `PkBkCalculator`).
            multipoles (list): List of multipoles to compute the model for.
        """
        self.priors = priors
        self.data = data
        self.calculator = calculator
        self.multipoles = multipoles

    def compute_model_vector(self, theta):
        """
        Compute the model predictions for the power spectrum and bispectrum based on the input parameters.

        Args:
            theta (np.ndarray): Array of parameter values sampled by the MCMC.

        Returns:
            np.ndarray: Concatenated model predictions for the specified multipoles.
        """
        # Convert the input array of MCMC points to a dictionary for the emulator
        parameters_to_vary = self.priors.copy()
        for name, value in zip(parameters_to_vary.keys(), theta):
            parameters_to_vary[name] = value

        # Initialize an empty list to store the model predictions
        model_vector = []

        # Separate multipoles into power spectrum (Pk) and bispectrum (Bk)
        multipoles_pk = {i for i in self.multipoles if len(i) == 1}
        multipoles_bk = {i for i in self.multipoles if len(i) == 3}

        # Compute power spectrum predictions
        if multipoles_pk:
            for L in multipoles_pk:
                k_from_data = self.data[L]['k']
                model_pk = self.calculator.pk_from_emulator(parameters_to_vary, L)(k_from_data)
                model_vector.append(model_pk)

        # Compute bispectrum predictions
        if multipoles_bk:
            for l1l2L in multipoles_bk:
                k_from_data = self.data[l1l2L]['k']
                model_bk = self.calculator.bk_from_emulator(parameters_to_vary, l1l2L)(k_from_data)
                model_vector.append(model_bk)

        # Concatenate the model predictions into a single array
        return np.concatenate(model_vector)
