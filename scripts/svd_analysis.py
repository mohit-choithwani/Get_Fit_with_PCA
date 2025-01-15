import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt

class ExerciseAnalyzer:
    def __init__(self, data):
        """
        Initialize the ExerciseAnalyzer with data.

        Args:
            data (np.ndarray): The input data (e.g., landmarks) with shape (samples, features).
        """
        self.data = data
        self.mean = np.mean(data, axis=0)
        self.mean_free_data = data - self.mean
        self.u, self.B, self.v = self._perform_svd(self.mean_free_data)
        self.covariance = np.cov(data, rowvar=False)

    def _perform_svd(self, data):
        """
        Perform SVD on the mean-free data.

        Args:
            data (np.ndarray): Mean-free data matrix.

        Returns:
            tuple: u, B (singular value matrix), v
        """

        u, s, v = np.linalg.svd(data.T, full_matrices=False)  # Ensure reduced form
        B = np.diag(s)  # Create a square diagonal matrix
        return u, B, v

    def generate_sample(self, parameters):
        """
        Generate a sample in the original space from given parameters.

        Args:
            parameters (np.ndarray): Parameters in PCA space.

        Returns:
            np.ndarray: Sample in the original space.
        """
        rows = self.mean_free_data.shape[0]
        D_pow_half = self.B / np.sqrt(rows)
        sample_wo_param = self.u @ D_pow_half.T
        sample = sample_wo_param @ parameters
        return sample.flatten() + self.mean

    # def generate_parameters(self, sample):
    #     """
    #     Generate parameters for a given sample in the PCA space.

    #     Args:
    #         sample (np.ndarray): A sample in the original space.

    #     Returns:
    #         np.ndarray: Parameters in PCA space.
    #     """
    #     rows = self.mean_free_data.shape[0]
    #     D_pow_half = self.B / np.sqrt(rows)
    #     D_pow_half_inv = np.linalg.pinv(D_pow_half)
    #     u_inv = np.linalg.inv(self.u)
    #     sample = sample.reshape((-1, 1))
    #     print("Shape of D_pow_half_inv.T:", D_pow_half_inv.T.shape)
    #     print("Shape of u_inv:", u_inv.shape)
    #     print("Shape of sample:", sample.shape)
        
    #     parameters = D_pow_half_inv.T @ u_inv @ sample
    #     return parameters.flatten()
    
    def generate_parameters(self, data):
        rows, columns = data.shape
        D_pow_half = self.B / np.sqrt(rows)
        D_pow_half_inv = np.linalg.pinv(D_pow_half)

        # Ensure data is aligned to the feature space
        assert data.shape[1] == self.u.shape[0], "Data and U matrix dimensions mismatch!"
        
        u_inv = np.linalg.inv(self.u)
        parameters = D_pow_half_inv.T @ u_inv @ data.T
        return parameters.T

    def check_gaussian(self, parameters):
        """
        Check if the parameters follow a Gaussian distribution.

        Args:
            parameters (np.ndarray): Parameters to test.

        Returns:
            bool: True if Gaussian, False otherwise.
        """
        stat, p = shapiro(parameters)
        alpha = 0.05
        is_gaussian = p > alpha
        return is_gaussian

    def predict_correctness(self, sample):
        """
        Predict if the given sample represents a correct exercise.

        Args:
            sample (np.ndarray): A sample in the original space.

        Returns:
            bool: True if the exercise is correct, False otherwise.
        """
        parameters = self.generate_parameters(sample)
        positive_params = parameters[parameters != 0]
        return self.check_gaussian(positive_params)
    
    def get_metadata(self):
        return (self.mean, self.u, self.B, self.v, self.covariance)