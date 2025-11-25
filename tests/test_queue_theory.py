"""
Tests para la clase MMOneQueue (métricas analíticas)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.queue_theory import MMOneQueue


def test_mm1_basic_metrics():
    """
    Test 1: Verificar métricas M/M/1 básicas con parámetros conocidos

    Sistema M/M/1 simple (sin heterogeneidad):
    λ = 3, μ = 4
    ρ = λ/μ = 0.75
    L = ρ/(1-ρ) = 3
    W = 1/(μ-λ) = 1
    Lq = ρ²/(1-ρ) = 2.25
    Wq = ρ/(μ-λ) = 0.75
    """
    print("\n=== Test 1: Métricas M/M/1 Básicas ===")

    lambda_rate = 3.0
    mu = 4.0
    pi = [1.0, 0.0]  # 100% un solo tipo

    queue = MMOneQueue(lambda_rate, mu, mu, pi)
    metrics = queue.calculate_metrics()

    print(f"Parámetros: λ={lambda_rate}, μ={mu}")
    print(f"Métricas calculadas: {metrics}")

    # Valores esperados
    expected = {"rho": 0.75, "L": 3.0, "W": 1.0, "Lq": 2.25, "Wq": 0.75}

    print(f"\nValores esperados: {expected}")

    # Verificar cada métrica
    for key in ["rho", "L", "W", "Lq", "Wq"]:
        calculated = metrics[key]
        expected_val = expected[key]
        error = abs(calculated - expected_val)
        print(
            f"{key}: calculado={calculated:.6f}, esperado={expected_val:.6f}, error={error:.6e}"
        )
        assert error < 1e-6, (
            f"{key} incorrecto: esperado {expected_val}, obtenido {calculated}"
        )

    print("\nTest 1 PASADO\n")


def test_weighted_service_rate():
    """
    Test 2: Verificar cálculo de μ efectiva con dos tipos de turistas

    Con π = [0.625, 0.375], μ_p = 6, μ_i = 4
    E[S] = 0.625*(1/6) + 0.375*(1/4) = 0.625*0.1667 + 0.375*0.25
         = 0.1042 + 0.0938 = 0.1979
    μ_eff = 1/E[S] = 1/0.1979 = 5.052
    """
    print("\n=== Test 2: Tasa de Servicio Ponderada ===")

    lambda_rate = 5.0
    mu_p = 6.0
    mu_i = 4.0
    pi = [0.625, 0.375]  # Distribución estacionaria del problema

    queue = MMOneQueue(lambda_rate, mu_p, mu_i, pi)

    E_S = pi[0] * (1 / mu_p) + pi[1] * (1 / mu_i)
    mu_eff_expected = 1 / E_S

    print(f"π = {pi}")
    print(f"μ_paciente = {mu_p}, μ_impaciente = {mu_i}")
    print(f"E[S] calculado = {E_S:.6f}")
    print(f"μ_eff esperado = {mu_eff_expected:.6f}")
    print(f"μ_eff obtenido = {queue.mu_eff:.6f}")

    error = abs(queue.mu_eff - mu_eff_expected)
    print(f"Error: {error:.6e}")

    assert error < 1e-6, f"μ_eff incorrecto"
    print("μ_eff calculado correctamente")

    print("\nTest 2 PASADO\n")


def test_system_stability():
    """
    Test 3: Verificar que el sistema detecta inestabilidad cuando ρ ≥ 1
    """
    print("\n=== Test 3: Estabilidad del Sistema ===")

    # Caso 1: Sistema estable (ρ < 1)
    print("\nCaso 1: Sistema estable (λ=3, μ=5)")
    lambda_rate = 3.0
    mu = 5.0
    pi = [1.0, 0.0]

    queue = MMOneQueue(lambda_rate, mu, mu, pi)
    metrics = queue.calculate_metrics()

    print(f"ρ = {metrics['rho']:.4f}")
    assert metrics is not None, "Sistema estable debe retornar métricas"
    assert metrics["rho"] < 1, "ρ debe ser < 1 para sistema estable"
    print("Sistema estable detectado correctamente")

    # Caso 2: Sistema inestable (ρ = 1)
    print("\nCaso 2: Sistema inestable (λ=5, μ=5)")
    lambda_rate = 5.0
    mu = 5.0

    queue = MMOneQueue(lambda_rate, mu, mu, pi)
    metrics = queue.calculate_metrics()

    print(f"ρ = {queue.rho:.4f}")
    assert metrics is None, "Sistema con ρ ≥ 1 debe retornar None"
    print("Sistema inestable detectado correctamente")

    # Caso 3: Sistema inestable (ρ > 1)
    print("\nCaso 3: Sistema inestable (λ=6, μ=5)")
    lambda_rate = 6.0
    mu = 5.0

    queue = MMOneQueue(lambda_rate, mu, mu, pi)
    metrics = queue.calculate_metrics()

    print(f"ρ = {queue.rho:.4f}")
    assert metrics is None, "Sistema con ρ > 1 debe retornar None"
    print("Sistema inestable detectado correctamente")

    print("\nTest 3 PASADO\n")


def test_problem_parameters():
    """
    Test 4: Verificar métricas con los parámetros exactos del problema

    λ = 5, μ_p = 6, μ_i = 4, π = [0.625, 0.375]
    """
    print("\n=== Test 4: Parámetros del Problema ===")

    lambda_rate = 5.0
    mu_p = 6.0
    mu_i = 4.0
    pi = [0.625, 0.375]

    queue = MMOneQueue(lambda_rate, mu_p, mu_i, pi)
    metrics = queue.calculate_metrics()

    print(f"Parámetros del problema:")
    print(f"  λ = {lambda_rate}")
    print(f"  μ_paciente = {mu_p}")
    print(f"  μ_impaciente = {mu_i}")
    print(f"  π = {pi}")
    print(f"\nMétricas calculadas:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Verificaciones básicas
    assert metrics["rho"] < 1, "Sistema debe ser estable"
    print("OK: Sistema estable (ρ < 1)")

    assert metrics["L"] > 0, "L debe ser positivo"
    assert metrics["W"] > 0, "W debe ser positivo"
    assert metrics["Lq"] >= 0, "Lq debe ser no negativo"
    assert metrics["Wq"] >= 0, "Wq debe ser no negativo"
    print("OK: Todas las métricas son válidas")

    # Verificar relaciones de Little
    # L = λ * W
    L_from_little = lambda_rate * metrics["W"]
    error_L = abs(metrics["L"] - L_from_little)
    print(f"\nLey de Little: L = λ·W")
    print(f"  L calculado = {metrics['L']:.6f}")
    print(f"  λ·W = {L_from_little:.6f}")
    print(f"  Error = {error_L:.6e}")
    assert error_L < 1e-6, "Debe cumplir L = λ·W"
    print("OK: Cumple L = λ·W")

    # Lq = λ * Wq
    Lq_from_little = lambda_rate * metrics["Wq"]
    error_Lq = abs(metrics["Lq"] - Lq_from_little)
    print(f"\nLey de Little: Lq = λ·Wq")
    print(f"  Lq calculado = {metrics['Lq']:.6f}")
    print(f"  λ·Wq = {Lq_from_little:.6f}")
    print(f"  Error = {error_Lq:.6e}")
    assert error_Lq < 1e-6, "Debe cumplir Lq = λ·Wq"
    print("OK: Cumple Lq = λ·Wq")

    print("\n[PASS] Test 4 PASADO\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Ejecutando tests para MMOneQueue")
    print("=" * 60)

    try:
        test_mm1_basic_metrics()
        test_weighted_service_rate()
        test_system_stability()
        test_problem_parameters()

        print("\n" + "=" * 60)
        print("[PASS] TODOS LOS TESTS DE QUEUE THEORY PASARON")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[FAIL] TEST FALLÓ: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] ERROR INESPERADO: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
