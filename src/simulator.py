# src/simulator.py - Simulación de eventos discretos M/M/1 con Cadena de Markov
import numpy as np
import pandas as pd


class TouristCenterSimulator:
    """
    Simulador de cola M/M/1 con tipos de turistas según cadena de Markov.

    Proceso de simulación:
    1. Generamos llegadas según proceso Poisson (λ)
    2. Cada turista tiene tipo según cadena de Markov
    3. Tiempo de servicio depende del tipo (μ_paciente o μ_impaciente)
    4. Sistema FIFO: un servidor, cola ilimitada
    """

    def __init__(self, lambda_rate, mu_patient, mu_impatient, markov_chain):
        self.lambda_rate = lambda_rate
        self.mu_p = mu_patient
        self.mu_i = mu_impatient
        self.markov = markov_chain

    def simulate(self, n_tourists=1000):
        """
        Simula n_tourists llegadas al sistema usando simulación de eventos discretos.

        Returns:
            DataFrame con información de cada turista y métricas del sistema
        """
        # Estado inicial de la cadena de Markov (según distribución estacionaria)
        current_state = np.random.choice([0, 1], p=self.markov.pi)

        # Generar TODAS las llegadas usando proceso Poisson
        inter_arrival_times = np.random.exponential(1 / self.lambda_rate, n_tourists)
        arrival_times = np.cumsum(inter_arrival_times)

        # Generar tipos de turistas según cadena de Markov
        tourist_types = []
        tourist_states = []

        for i in range(n_tourists):
            tourist_states.append(current_state)
            tourist_types.append(self.markov.states[current_state])
            current_state = self.markov.next_state(current_state)

        # Generar tiempos de servicio según tipo
        service_times = []
        for state in tourist_states:
            mu = self.mu_p if state == 0 else self.mu_i
            service_times.append(np.random.exponential(1 / mu))

        # Simulación de eventos discretos M/M/1
        service_start_times = []
        departure_times = []
        wait_times = []

        server_free_at = 0  # Momento en que el servidor se libera

        for i in range(n_tourists):
            arrival_time = arrival_times[i]
            service_time = service_times[i]

            # El servicio comienza cuando el turista llega O cuando el servidor se libera
            service_start = max(arrival_time, server_free_at)
            wait_time = service_start - arrival_time
            departure = service_start + service_time

            service_start_times.append(service_start)
            departure_times.append(departure)
            wait_times.append(wait_time)

            # Actualizar cuando el servidor estará libre
            server_free_at = departure

        # Calcular tiempo en sistema para cada turista
        time_in_system = [
            departure_times[i] - arrival_times[i] for i in range(n_tourists)
        ]

        # Calcular métricas usando Ley de Little
        # Tiempo total de simulación
        total_simulation_time = departure_times[-1] - arrival_times[0]

        # Tasa de llegadas empírica
        lambda_empirical = n_tourists / total_simulation_time

        # Tiempos promedio
        W_simulated = np.mean(time_in_system)
        Wq_simulated = np.mean(wait_times)

        # Aplicar Ley de Little: L = λ * W
        L_simulated = lambda_empirical * W_simulated
        Lq_simulated = lambda_empirical * Wq_simulated

        # Calcular N(t) como función del tiempo continuo
        # Crear eventos ordenados por tiempo: llegadas (+1) y salidas (-1)
        events = []
        for i in range(n_tourists):
            events.append((arrival_times[i], +1, i))  # llegada
            events.append((departure_times[i], -1, i))  # salida

        # Ordenar eventos por tiempo (salidas antes que llegadas si mismo tiempo)
        events.sort(key=lambda x: (x[0], x[1]))

        # Calcular N(t) y área bajo la curva para L promedio
        n_current = 0
        prev_time = 0
        total_area = 0  # Integral de N(t) dt
        total_queue_area = 0  # Integral de max(N(t)-1, 0) dt

        # También guardar N(t) en cada llegada para el DataFrame
        n_at_arrival = np.zeros(n_tourists, dtype=int)

        # Procesar eventos
        for time, delta, tourist_idx in events:
            # Acumular área antes del cambio
            dt = time - prev_time
            total_area += n_current * dt
            total_queue_area += max(n_current - 1, 0) * dt

            # Si es una llegada, guardar N(t) justo después de llegar
            if delta == +1:
                n_current += 1
                n_at_arrival[tourist_idx] = n_current
            else:
                n_current -= 1

            prev_time = time

        # Calcular L y Lq usando el método de área (más preciso)
        total_time = departure_times[-1] - arrival_times[0]
        L_by_area = total_area / total_time
        Lq_by_area = total_queue_area / total_time

        # Crear DataFrame con información por turista
        results = pd.DataFrame(
            {
                "turista": range(1, n_tourists + 1),
                "tiempo_llegada": arrival_times,
                "tiempo_inicio_servicio": service_start_times,
                "tiempo_salida": departure_times,
                "tipo": tourist_types,
                "estado": tourist_states,
                "tiempo_servicio": service_times,
                "tiempo_espera": wait_times,
                "tiempo_en_sistema": time_in_system,
                "n_sistema": n_at_arrival,
            }
        )

        # Guardar métricas calculadas
        # Usar método de área (más preciso que Ley de Little para simulaciones finitas)
        results.attrs["L_simulated"] = L_by_area
        results.attrs["W_simulated"] = W_simulated
        results.attrs["Wq_simulated"] = Wq_simulated
        results.attrs["Lq_simulated"] = Lq_by_area
        results.attrs["lambda_empirical"] = lambda_empirical
        results.attrs["total_time"] = total_time

        # También guardar versiones por Ley de Little para comparación
        results.attrs["L_little"] = L_simulated
        results.attrs["Lq_little"] = Lq_simulated

        # Estado estacionario (últimos 20% de turistas)
        steady_state_start = int(0.8 * n_tourists)
        results.attrs["N_steady_state"] = results["n_sistema"][
            steady_state_start:
        ].mean()

        return results
