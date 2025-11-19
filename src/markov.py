import numpy as np


class MarkovChain:
    """Cadena de Markov para tipos de turistas"""

    def __init__(self, transition_matrix):
        """
        Args:
            transition_matrix: Matriz P (numpy array)
        """
        self.P = np.array(transition_matrix)
        self.states = ["Paciente", "Impaciente"]
        self.pi = self._calculate_stationary()

    def _calculate_stationary(self):
        """Calcula distribución estacionaria π"""
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        stationary = stationary / stationary.sum()
        return stationary.flatten()

    def next_state(self, current_state):
        """
        Genera siguiente estado según probabilidades
        Args:
            current_state: 0 (Paciente) o 1 (Impaciente)
        Returns:
            Siguiente estado (0 o 1)
        """
        return np.random.choice([0, 1], p=self.P[current_state])

    def get_stationary_distribution(self):
        """Retorna π como dict"""
        return {"Paciente": self.pi[0], "Impaciente": self.pi[1]}
