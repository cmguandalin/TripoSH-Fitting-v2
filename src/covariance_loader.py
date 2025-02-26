import numpy as np

class CovarianceLoader:
    def __init__(self, cov_path, multipoles, k_edges, rescale=None):
        """
        Initialize the CovarianceLoader class.

        Args:
            cov_path (str): Path to the covariance file.
            multipoles (list): List of multipoles to include (e.g., ['0', '2', '000']).
            k_edges (dict): Dictionary mapping multipoles to their k-range edges.
            rescale (float, optional): Rescaling factor for the covariance matrix. Defaults to None.
        """
        self.cov_path   = cov_path
        self.multipoles = multipoles
        self.k_edges    = k_edges
        self.rescale    = rescale
        self.covariance = None
        self.final_covariance = None

    def load_covariance(self):
        """
        Load the covariance matrix from the specified path.
        """
        self.covariance = np.load(self.cov_path, allow_pickle=True).item()

    def filter_multipoles(self):
        """
        Filter the covariance matrix to include only the specified multipoles.
        """
        indices_to_keep = []
        current_index = 0

        for ell in self.covariance['k']:
            if ell in self.multipoles:
                indices_to_keep.extend(range(current_index, current_index + len(self.covariance['k'][ell])))
            current_index += len(self.covariance['k'][ell])

        self.covariance['cov'] = self.covariance['cov'][np.ix_(indices_to_keep, indices_to_keep)]

        # Remove multipoles not used in the analysis
        self.covariance['k'] = {ell: self.covariance['k'][ell] for ell in self.multipoles if ell in self.covariance['k']}

    def filter_wavemodes(self):
        """
        Filter the covariance matrix to include only wavemodes within the specified k-range.
        """
        mask_wavemodes = np.concatenate([
            (self.k_edges[ell][0] <= self.covariance['k'][ell]) & (self.k_edges[ell][1] >= self.covariance['k'][ell])
            for ell in self.multipoles
        ])

        self.final_covariance = self.covariance['cov'][mask_wavemodes, :][:, mask_wavemodes]

        # Rescale the covariance matrix if a rescale factor is provided
        if self.rescale:
            self.final_covariance *= self.rescale

    def get_covariance(self):
        """
        Return the final processed covariance matrix.

        Returns:
            np.ndarray: The final covariance matrix.
        """
        return self.final_covariance

    def process(self):
        """
        Execute the full processing pipeline: load, filter multipoles, and filter wavemodes.
        """
        self.load_covariance()
        self.filter_multipoles()
        self.filter_wavemodes()
