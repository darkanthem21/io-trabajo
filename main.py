from asyncio.tasks import run_coroutine_threadsafe
from profile import run

import matplotlib.pyplot as plt

from src.gui import run_gui
from src.markov import MarkovChain
from src.queue_theory import MMOneQueue
from src.simulator import TouristCenterSimulator


def main():
    # Parámetros
    lambda_rate = 5
    mu_patient = 6
    mu_impatient = 4
    P = [[0.7, 0.3], [0.5, 0.5]]

    # Cadena de Markov
    markov = MarkovChain(P)
    pi_dist = markov.get_stationary_distribution()
    print("Distribución estacionaria π:")
    print(f"  Paciente: {pi_dist['Paciente']:.4f}")
    print(f"  Impaciente: {pi_dist['Impaciente']:.4f}\n")

    # Métricas analíticas
    queue = MMOneQueue(lambda_rate, mu_patient, mu_impatient, markov.pi)
    metrics = queue.calculate_metrics()
    print("Métricas M/M/1 ponderadas:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Simulación
    n_tourists = 1000
    print(f"\nSimulando {n_tourists} turistas...")
    simulator = TouristCenterSimulator(lambda_rate, mu_patient, mu_impatient, markov)
    results = simulator.simulate(n_tourists)

    # Métricas simuladas
    L_sim = results.attrs.get("L_simulated", 0)
    W_sim = results.attrs.get("W_simulated", 0)
    Lq_sim = results.attrs.get("Lq_simulated", 0)
    Wq_sim = results.attrs.get("Wq_simulated", 0)

    print(f"\nMétricas simuladas:")
    print(f"  L (clientes en sistema): {L_sim:.2f}")
    print(f"  W (tiempo en sistema): {W_sim:.2f} min")
    print(f"  Lq (clientes en cola): {Lq_sim:.2f}")
    print(f"  Wq (tiempo en cola): {Wq_sim:.2f} min")

    # Comparación teórico vs simulado
    print(f"\nComparación teórico vs simulado:")
    print(
        f"  L:  teórico={metrics['L']:.2f}, simulado={L_sim:.2f}, error={(abs(metrics['L'] - L_sim) / metrics['L'] * 100):.1f}%"
    )
    print(
        f"  W:  teórico={metrics['W']:.2f}, simulado={W_sim:.2f}, error={(abs(metrics['W'] - W_sim) / metrics['W'] * 100):.1f}%"
    )
    print(
        f"  Lq: teórico={metrics['Lq']:.2f}, simulado={Lq_sim:.2f}, error={(abs(metrics['Lq'] - Lq_sim) / metrics['Lq'] * 100):.1f}%"
    )
    print(
        f"  Wq: teórico={metrics['Wq']:.2f}, simulado={Wq_sim:.2f}, error={(abs(metrics['Wq'] - Wq_sim) / metrics['Wq'] * 100):.1f}%"
    )

    # Estadísticas
    print(f"\nEstadísticas adicionales:")
    print(f"  N(t) promedio en llegadas: {results['n_sistema'].mean():.2f}")
    print(
        f"  N(t) estado estacionario (últimos 20%): {results.attrs.get('N_steady_state', 0):.2f}"
    )
    print(f"  Distribución tipos:")
    print(results["tipo"].value_counts(normalize=True))

    # Gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(results["turista"], results["n_sistema"])
    plt.xlabel("Turista")
    plt.ylabel("N(t) - Turistas en sistema")
    plt.title("Evolución del sistema")
    plt.grid(True)
    plt.savefig("figures/n_t_evolution.png")
    print("\nGráfico guardado en figures/n_t_evolution.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        main()
    else:
        run_gui()
