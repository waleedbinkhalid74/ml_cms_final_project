import numpy as np
from scipy.linalg import svd


class DMD:
    def __init__(self, data: np.array, time: np.array, transpose: bool = False):
        """
        Initializes the Dynamic Mode Decomposition class object with training data.
        Data should be structured such that each column represents a state in the given time step.
        If the data is transposed, the parameter transpose must be set to True.

        Args:
            data (2 dim np.array): Input data with column structure
                State_at_time_0 State_at_time_1 ... State_at_time_m
            time (1 dim np.array): array representing the time points of the measurements
            transpose (bool): [Optional] a flag to transpose the data matrix.
        """
        self.data = data.T if transpose else data
        self.X0 = self.data[:, :-1]
        self.X1 = self.data[:, 1:]
        self.t = time
        self.dt = self.t[1] - self.t[0]
        self.eig_vals = None
        self.phi = None
        self.r = len(self.t)

    def get_modes(self, modes: np.int32 = np.iinfo(np.int32).max):
        """
        Calculates the DMD modes.
        Args:
            modes: The number of DMD modes to calculate.
                The number of modes is the minimum of the rank of the data and the given number.
        Returns:
            2 dim np.array: DMD modes
        """
        self.r = min(modes, np.linalg.matrix_rank(self.data))

        u, s, vh = svd(self.X0, full_matrices=False)
        v = vh.conj().T
        u_r = u[:, :self.r]
        v_r = v[:, 0:self.r]
        inv_s = np.diag(np.reciprocal(s[:self.r]))

        A_tilda = u_r.conj().T @ self.X1 @ v_r @ inv_s

        self.eig_vals, eig_vectors = np.linalg.eig(A_tilda)

        self.phi = self.X1 @ v_r @ inv_s @ eig_vectors

        return self.phi

    def reconstruct(self):
        """
        Reconstruct the data using the calculated modes.
        Args:
        Returns:
            2 dim np.array: reconstructed data using DMD,
            1 dim np.array: the coefficients of
        """
        if self.phi is None:
            self.get_modes()

        b = np.dot(np.linalg.pinv(self.phi), self.data[:, 0])
        time_dynamics = np.zeros((self.r, len(self.t)), dtype=self.data.dtype)

        for i, time_step in enumerate(self.t):
            time_dynamics[:, i] = b * np.power(self.eig_vals, time_step / self.dt)

        data_dmd = self.phi @ time_dynamics

        return data_dmd

    def predict(self, timesteps: int):
        """
        Reconstruct the data using the calculated modes.
        Args:
            timesteps (int): number of future time steps to predict
        Returns:
            2 dim np.array: reconstructed data using DMD,
            1 dim np.array: the coefficients of
        """
        if self.phi is None:
            self.get_modes()

        b = np.linalg.pinv(self.phi) @ self.data[:, 0]
        prediction = np.zeros((self.r, timesteps), dtype=self.data.dtype)

        for i in range(timesteps):
            prediction[:, i] = b * np.power(self.eig_vals, (len(self.t) + i) * self.dt)

        predicated_data = self.phi @ prediction

        return predicated_data
