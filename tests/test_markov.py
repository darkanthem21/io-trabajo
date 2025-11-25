import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.markov import MarkovChain


def test_stationary_distribution():
    """
    Test 1: Verificar que π estacionaria cumple π·P = π

    Con P = [[0.7, 0.3], [0.5, 0.5]]
    La distribución estacionaria debe cumplir:
    - π[0] + π[1] = 1 (suma = 1)
    - π·P = π (propiedad estacionaria)
    """
    print("\n=== Test 1: Distribución Estacionaria ===")

    # Matriz de transición del problema
    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    # Obtener π
    pi = markov.pi
    print(f"π calculada: {pi}")

    # Verificación 1: La suma debe ser 1
    suma = np.sum(pi)
    print(f"Suma de π: {suma:.6f}")
    assert abs(suma - 1.0) < 1e-6, f"La suma de π debe ser 1, pero es {suma}"
    print("OK: Suma de π = 1")

    # Verificación 2: π·P = π (propiedad estacionaria)
    pi_P = np.dot(pi, markov.P)
    print(f"π·P calculado: {pi_P}")
    diferencia = np.abs(pi_P - pi)
    print(f"Diferencia |π·P - π|: {diferencia}")
    assert np.all(diferencia < 1e-6), f"π·P debe ser igual a π"
    print("OK: π·P = π (propiedad estacionaria cumplida)")
    expected_pi = np.array([0.625, 0.375])
    print(f"π esperada: {expected_pi}")
    error = np.abs(pi - expected_pi)
    print(f"Error: {error}")
    assert np.all(error < 1e-6), f"π debe ser aproximadamente {expected_pi}"
    print("OK: Valores de π correctos")

    print("\n[PASS] Test 1\n")


def test_markov_states():
    """
    Test 2: Verificar que los estados se generan correctamente
    """
    print("\n=== Test 2: Generación de Estados ===")

    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    # Generar 1000 transiciones desde estado 0
    np.random.seed(42)  # Para reproducibilidad
    transitions = []
    current = 0
    for _ in range(1000):
        next_state = markov.next_state(current)
        transitions.append(next_state)
        current = next_state

    # Contar frecuencias
    unique, counts = np.unique(transitions, return_counts=True)
    frequencies = counts / len(transitions)

    print(f"Estados generados: {len(transitions)}")
    print(f"Frecuencias observadas: {dict(zip(unique, frequencies))}")

    # Los estados deben ser 0 o 1 únicamente
    assert set(unique).issubset({0, 1}), "Los estados deben ser 0 o 1"
    print("OK: Estados válidos (0 o 1)")

    # Con muchas transiciones, la frecuencia debe acercarse a π
    # (no exacto porque depende de estado inicial, pero debe estar cerca)
    print(f"π estacionaria para comparación: {markov.pi}")

    print("\n[PASS] Test 2\n")


def test_get_stationary_distribution():
    """
    Test 3: Verificar método get_stationary_distribution()
    """
    print("\n=== Test 3: Método get_stationary_distribution() ===")

    P = [[0.7, 0.3], [0.5, 0.5]]
    markov = MarkovChain(P)

    pi_dict = markov.get_stationary_distribution()

    print(f"Distribución como dict: {pi_dict}")

    # Verificar que retorna un diccionario
    assert isinstance(pi_dict, dict), "Debe retornar un diccionario"
    print("OK: Retorna un diccionario")

    # Verificar claves
    assert "Paciente" in pi_dict, "Debe contener clave 'Paciente'"
    assert "Impaciente" in pi_dict, "Debe contener clave 'Impaciente'"
    print("OK: Contiene claves 'Paciente' e 'Impaciente'")

    # Verificar suma = 1
    suma = pi_dict["Paciente"] + pi_dict["Impaciente"]
    assert abs(suma - 1.0) < 1e-6, "La suma debe ser 1"
    print(f"OK: Suma = {suma:.6f}")

    print("\n[PASS] Test 3\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Ejecutando tests para MarkovChain")
    print("=" * 60)

    try:
        test_stationary_distribution()
        test_markov_states()
        test_get_stationary_distribution()

        print("\n" + "=" * 60)
        print("RESULTADO: TODOS LOS TESTS DE MARKOV PASARON")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[FAIL] TEST FALLO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] ERROR INESPERADO: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
