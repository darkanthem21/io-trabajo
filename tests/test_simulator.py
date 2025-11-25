import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.markov import MarkovChain
from src.simulator import TouristCenterSimulator


def test_simulation_structure():
    """
    Test 1: Verificar estructura básica de la simulación
    """
    print("\n=== Test 1: Estructura de la Simulación ===")

    lambda_rate = 5.0
    mu_p = 6.0
    mu_i = 4.0
    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    simulator = TouristCenterSimulator(lambda_rate, mu_p, mu_i, markov)

    # Simular pocos turistas para el test
    n_tourists = 100
    results = simulator.simulate(n_tourists)

    print(f"Simulación de {n_tourists} turistas")
    print(f"Tipo de resultado: {type(results)}")

    # Verificar que retorna un DataFrame
    assert isinstance(results, pd.DataFrame), "Debe retornar un DataFrame"
    print("Retorna un DataFrame")

    # Verificar número de filas
    assert len(results) == n_tourists, f"Debe tener {n_tourists} filas"
    print(f"Tiene {n_tourists} filas")

    # Verificar columnas requeridas
    required_columns = [
        "turista",
        "tiempo_llegada",
        "tiempo_inicio_servicio",
        "tiempo_salida",
        "tipo",
        "estado",
        "tiempo_servicio",
        "tiempo_espera",
        "tiempo_en_sistema",
        "n_sistema",
    ]

    for col in required_columns:
        assert col in results.columns, f"Falta columna '{col}'"
    print(f"Tiene todas las columnas requeridas: {required_columns}")

    # Verificar atributos con métricas
    assert "L_simulated" in results.attrs, "Falta atributo 'L_simulated'"
    assert "W_simulated" in results.attrs, "Falta atributo 'W_simulated'"
    assert "Lq_simulated" in results.attrs, "Falta atributo 'Lq_simulated'"
    assert "Wq_simulated" in results.attrs, "Falta atributo 'Wq_simulated'"
    print("Tiene todos los atributos de métricas simuladas")

    print("\nTest 1 PASADO\n")


def test_tourist_types():
    """
    Test 2: Verificar que los tipos de turistas siguen la cadena de Markov
    """
    print("\n=== Test 2: Tipos de Turistas ===")

    lambda_rate = 5.0
    mu_p = 6.0
    mu_i = 4.0
    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    simulator = TouristCenterSimulator(lambda_rate, mu_p, mu_i, markov)

    # Simular muchos turistas para verificar distribución
    np.random.seed(42)
    n_tourists = 10000
    results = simulator.simulate(n_tourists)

    # Contar tipos
    type_counts = results["tipo"].value_counts()
    type_freq = results["tipo"].value_counts(normalize=True)

    print(f"Simulación de {n_tourists} turistas")
    print(f"Conteo de tipos: {type_counts.to_dict()}")
    print(f"Frecuencias: {type_freq.to_dict()}")

    # Verificar que solo hay dos tipos
    assert set(results["tipo"].unique()) == {"Paciente", "Impaciente"}, (
        "Solo debe haber tipos 'Paciente' e 'Impaciente'"
    )
    print("Solo hay tipos 'Paciente' e 'Impaciente'")

    # Con muchos turistas, las frecuencias deben acercarse a π
    # π = [0.625, 0.375] para esta matriz
    pi_expected = markov.get_stationary_distribution()

    freq_paciente = type_freq.get("Paciente", 0)
    freq_impaciente = type_freq.get("Impaciente", 0)

    print(f"\nDistribución estacionaria esperada: {pi_expected}")
    print(
        f"Frecuencia Paciente: {freq_paciente:.4f} (esperado: {pi_expected['Paciente']:.4f})"
    )
    print(
        f"Frecuencia Impaciente: {freq_impaciente:.4f} (esperado: {pi_expected['Impaciente']:.4f})"
    )

    # Con 10000 muestras, debería estar dentro de ±5%
    error_p = abs(freq_paciente - pi_expected["Paciente"])
    error_i = abs(freq_impaciente - pi_expected["Impaciente"])

    print(f"Error Paciente: {error_p:.4f}")
    print(f"Error Impaciente: {error_i:.4f}")

    assert error_p < 0.05, f"Frecuencia de Paciente muy alejada de π"
    assert error_i < 0.05, f"Frecuencia de Impaciente muy alejada de π"
    print("Frecuencias cercanas a distribución estacionaria")

    # Verificar estados (0 o 1)
    assert set(results["estado"].unique()).issubset({0, 1}), "Estados deben ser 0 o 1"
    print("Estados son 0 o 1")

    print("Test 2 PASADO\n")


def test_fifo_queue():
    """
    Test 3: Verificar que la cola funciona como FIFO
    """
    print("\n=== Test 3: Cola FIFO ===")

    lambda_rate = 5.0
    mu_p = 6.0
    mu_i = 4.0
    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    np.random.seed(123)
    simulator = TouristCenterSimulator(lambda_rate, mu_p, mu_i, markov)
    results = simulator.simulate(50)

    print("Verificando orden FIFO...")

    # En FIFO, el orden de inicio de servicio debe ser el mismo que el de llegada
    # (el que llega primero, se atiende primero)
    for i in range(len(results) - 1):
        llegada_i = results.iloc[i]["tiempo_llegada"]
        llegada_i1 = results.iloc[i + 1]["tiempo_llegada"]
        inicio_i = results.iloc[i]["tiempo_inicio_servicio"]
        inicio_i1 = results.iloc[i + 1]["tiempo_inicio_servicio"]

        # Si turista i llega antes que i+1, debe iniciar servicio antes o al mismo tiempo
        if llegada_i < llegada_i1:
            assert inicio_i <= inicio_i1, (
                f"Violación de FIFO: turista {i} llega antes pero inicia servicio después"
            )

    print("Orden FIFO respetado")

    # Verificar que tiempo_espera = tiempo_inicio_servicio - tiempo_llegada
    for i, row in results.iterrows():
        espera_calculada = row["tiempo_inicio_servicio"] - row["tiempo_llegada"]
        espera_registrada = row["tiempo_espera"]
        error = abs(espera_calculada - espera_registrada)
        assert error < 1e-6, f"Tiempo de espera mal calculado para turista {i}"

    print("Tiempos de espera correctos")

    # Verificar que tiempo_en_sistema = tiempo_salida - tiempo_llegada
    for i, row in results.iterrows():
        sistema_calculado = row["tiempo_salida"] - row["tiempo_llegada"]
        sistema_registrado = row["tiempo_en_sistema"]
        error = abs(sistema_calculado - sistema_registrado)
        assert error < 1e-6, f"Tiempo en sistema mal calculado para turista {i}"

    print("Tiempos en sistema correctos")

    print("Test 3 PASADO\n")


def test_nt_calculation():
    """
    Test 4: Verificar cálculo de N(t) - número de turistas en sistema
    """
    print("\n=== Test 4: Cálculo de N(t) ===")

    lambda_rate = 5.0
    mu_p = 6.0
    mu_i = 4.0
    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    np.random.seed(456)
    simulator = TouristCenterSimulator(lambda_rate, mu_p, mu_i, markov)
    results = simulator.simulate(100)

    print("Verificando cálculo de N(t)...")

    # N(t) debe ser al menos 1 cuando un turista llega
    # (porque acaba de entrar al sistema)
    assert all(results["n_sistema"] >= 1), "N(t) debe ser al menos 1 al llegar"
    print("N(t) ≥ 1 para todos los turistas")

    # Verificar que N(t) es razonable
    max_n = results["n_sistema"].max()
    print(f"Máximo N(t) observado: {max_n}")

    # Para ρ ≈ 0.99, es posible tener colas largas, pero no infinitas
    assert max_n < 1000, "N(t) parece demasiado grande"
    print("N(t) en rango razonable")

    print("\nTest 4 PASADO\n")


def test_metrics_consistency():
    """
    Test 5: Verificar consistencia de métricas simuladas
    """
    print("\n=== Test 5: Consistencia de Métricas ===")

    lambda_rate = 5.0
    mu_p = 6.0
    mu_i = 4.0
    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    np.random.seed(789)
    simulator = TouristCenterSimulator(lambda_rate, mu_p, mu_i, markov)
    results = simulator.simulate(1000)

    # Obtener métricas
    L_sim = results.attrs.get("L_simulated", 0)
    W_sim = results.attrs.get("W_simulated", 0)
    Lq_sim = results.attrs.get("Lq_simulated", 0)
    Wq_sim = results.attrs.get("Wq_simulated", 0)
    lambda_emp = results.attrs.get("lambda_empirical", 0)

    print(f"Métricas simuladas:")
    print(f"  L = {L_sim:.4f}")
    print(f"  W = {W_sim:.4f}")
    print(f"  Lq = {Lq_sim:.4f}")
    print(f"  Wq = {Wq_sim:.4f}")
    print(f"  λ empírico = {lambda_emp:.4f}")

    # Verificar Ley de Little: L ≈ λ * W
    L_from_little = lambda_emp * W_sim
    error_L = abs(L_sim - L_from_little) / L_sim if L_sim > 0 else 0

    print(f"\nLey de Little: L = λ·W")
    print(f"  L simulado = {L_sim:.4f}")
    print(f"  λ·W = {L_from_little:.4f}")
    print(f"  Error relativo = {error_L:.4%}")

    # Debe estar muy cerca (método de área es preciso)
    assert error_L < 0.01, "L debe cumplir Ley de Little"
    print("Cumple L ≈ λ·W")

    # Verificar Ley de Little: Lq ≈ λ * Wq
    Lq_from_little = lambda_emp * Wq_sim
    error_Lq = abs(Lq_sim - Lq_from_little) / Lq_sim if Lq_sim > 0 else 0

    print(f"\nLey de Little: Lq = λ·Wq")
    print(f"  Lq simulado = {Lq_sim:.4f}")
    print(f"  λ·Wq = {Lq_from_little:.4f}")
    print(f"  Error relativo = {error_Lq:.4%}")

    assert error_Lq < 0.01, "Lq debe cumplir Ley de Little"
    print("Cumple Lq ≈ λ·Wq")

    # Verificar que L > Lq (siempre hay al menos el que está siendo atendido)
    assert L_sim >= Lq_sim, "L debe ser mayor o igual que Lq"
    print(f"OK: L ({L_sim:.4f}) ≥ Lq ({Lq_sim:.4f})")

    # Verificar que W > Wq (tiempo en sistema incluye tiempo de servicio)
    assert W_sim >= Wq_sim, "W debe ser mayor o igual que Wq"
    print(f"OK: W ({W_sim:.4f}) ≥ Wq ({Wq_sim:.4f})")

    print("\nTest 5 PASADO\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Ejecutando tests para TouristCenterSimulator")
    print("=" * 60)

    try:
        test_simulation_structure()
        test_tourist_types()
        test_fifo_queue()
        test_nt_calculation()
        test_metrics_consistency()

        print("\n" + "=" * 60)
        print("TODOS LOS TESTS DEL SIMULADOR PASARON")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n TEST FALLÓ: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n ERROR INESPERADO: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
