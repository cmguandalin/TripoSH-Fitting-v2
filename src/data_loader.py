import numpy as np

class DataLoader:
    def __init__(self, data_path, data_files, multipoles):
        self.data_path = data_path
        self.data_files = data_files
        self.multipoles = multipoles
        self.data = {}

    def load_data(self, k_edges):
        self.multipoles_pk = {i for i in self.multipoles if len(i) == 1} or None
        self.multipoles_bk = {i for i in self.multipoles if len(i) == 3} or None

        if self.multipoles_pk:
            print(f'Initialising power spectrum data from {self.data_path}.')
            for ell in self.multipoles_pk:
                print(f'{ell}: {self.data_files[ell]}')
                path_to_file = self.data_path + self.data_files[ell]
                tmp_k, tmp_pk = np.loadtxt(path_to_file, unpack=True)
                self.data[ell] = {}
                mask = (tmp_k >= min(k_edges[ell])) & (tmp_k <= max(k_edges[ell]))
                self.data[ell]['k'] = tmp_k[mask]
                self.data[ell]['Pk'] = tmp_pk[mask]
                
        if self.multipoles_bk:
            print(f'Initialising bispectrum data from {self.data_path}.')
            for ell in self.multipoles_bk:
                path_to_file = self.data_path + self.data_files[ell]
                print(f'{ell}: {self.data_files[ell]}')
                tmp_k, tmp_bk = np.loadtxt(path_to_file, unpack=True)
                self.data[ell] = {}
                mask = (tmp_k >= min(k_edges[ell])) & (tmp_k <= max(k_edges[ell]))
                self.data[ell]['k'] = tmp_k[mask]
                self.data[ell]['Bk'] = tmp_bk[mask]

    def get_data(self):
        return self.data

    def get_concatenated_data(self):
        """
        Automatically scan through multipoles in `data` and concatenate k and corresponding data values (Pk or Bk).
    
        Args:
            data (dict): Dictionary containing multipoles and their corresponding k and data values.
    
        Returns:
            tuple: A tuple containing:
                - full_k (np.ndarray): Concatenated k values.
                - full_data (np.ndarray): Concatenated data values (Pk or Bk).
        """
        self.full_k = np.array([])
        self.full_data = np.array([])
    
        #for ell, values in self.data.items():
        #    if 'Pk' in values:
        #        self.full_k = np.concatenate((self.full_k, values['k']))
        #        self.full_data = np.concatenate((self.full_data, values['Pk']))
        #    elif 'Bk' in values:
        #        self.full_k = np.concatenate((self.full_k, values['k']))
        #        self.full_data = np.concatenate((self.full_data, values['Bk']))
    
        #return self.full_k, self.full_data

        desired_order = ['0', '2', '4', '000', '202']
        
        # Initialize empty arrays for concatenation
        self.full_k = np.array([])
        self.full_data = np.array([])
        
        # Iterate through multipoles in the desired order
        for ell in desired_order:
            if ell in self.data:  # Check if the multipole exists in the data
                values = self.data[ell]
                if 'Pk' in values:  # Power spectrum multipole
                    self.full_k = np.concatenate((self.full_k, values['k']))
                    self.full_data = np.concatenate((self.full_data, values['Pk']))
                elif 'Bk' in values:  # Bispectrum multipole
                    self.full_k = np.concatenate((self.full_k, values['k']))
                    self.full_data = np.concatenate((self.full_data, values['Bk']))
        
        return self.full_k, self.full_data
