#!/usr/bin/env python3
"""
Script para ejecutar todos los tests del proyecto
"""

import os
import subprocess
import sys


def run_test(test_file, test_name):
    """Ejecuta un archivo de test y reporta el resultado"""
    print(f"\n{'=' * 70}")
    print(f"Ejecutando: {test_name}")
    print(f"{'=' * 70}")

    if os.path.exists("./venv/bin/python"):
        python_cmd = "./venv/bin/python"
    else:
        python_cmd = "python3"

    try:
        result = subprocess.run(
            [python_cmd, test_file], capture_output=False, text=True, check=True
        )
        print(f"\n[PASS] {test_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {test_name}")
        return False
    except FileNotFoundError:
        print(f"\n[WARN] {test_name} - No encontrado: {test_file}")
        return False


def main():
    """Ejecuta todos los tests"""

    tests = [
        ("tests/test_markov.py", "Test 1: Cadena de Markov"),
        ("tests/test_queue_theory.py", "Test 2: Teoría de Colas M/M/1"),
        ("tests/test_simulator.py", "Test 3: Simulador"),
        ("tests/test_integration.py", "Test 4: Integración Completa"),
    ]

    results = []

    for test_file, test_name in tests:
        passed = run_test(test_file, test_name)
        results.append((test_name, passed))

    # Resumen final
    print(f"\n\n{'=' * 70}")
    print("RESUMEN DE TESTS")
    print(f"{'=' * 70}\n")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {test_name}")

    print(f"\n{'=' * 70}")
    print(f"Total: {passed_count}/{total_count} tests pasaron")
    print(f"{'=' * 70}\n")

    if passed_count == total_count:
        print("RESULTADO: Todos los tests pasaron correctamente.\n")
        return 0
    else:
        print(
            f"[WARN] {total_count - passed_count} test(s) fallaron. Revisa los errores arriba.\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
