class MMOneQueue:
    """Métricas analíticas M/M/1 con tipos heterogéneos"""

    def __init__(self, lambda_rate, mu_patient, mu_impatient, pi):
        """
        Args:
            lambda_rate: Tasa de llegada
            mu_patient: Tasa servicio pacientes
            mu_impatient: Tasa servicio impacientes
            pi: [prob_paciente, prob_impaciente]
        """
        self.lambda_rate = lambda_rate
        self.mu_p = mu_patient
        self.mu_i = mu_impatient
        self.pi = pi

        # Tiempo de servicio promedio ponderado: E[S] = π_p * (1/μ_p) + π_i * (1/μ_i)
        # La tasa efectiva es el inverso del tiempo promedio
        avg_service_time = pi[0] * (1 / mu_patient) + pi[1] * (1 / mu_impatient)
        self.mu_eff = 1 / avg_service_time
        self.rho = lambda_rate / self.mu_eff

        # Mantener mu_avg para compatibilidad (aunque conceptualmente es mu_eff)
        self.mu_avg = self.mu_eff

    def calculate_metrics(self):
        """Calcula L, W, Lq, Wq"""
        if self.rho >= 1:
            return None  # sistema no es estable

        L = self.rho / (1 - self.rho)
        W = 1 / (self.mu_avg - self.lambda_rate)
        Lq = (self.rho**2) / (1 - self.rho)
        Wq = self.rho / (self.mu_avg - self.lambda_rate)

        return {
            "rho": self.rho,
            "L": L,
            "W": W,
            "Lq": Lq,
            "Wq": Wq,
            "mu_avg": self.mu_avg,
        }
