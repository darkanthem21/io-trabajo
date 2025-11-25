# Simulador M/M/1 con Cadena de Markov

**Proyecto:** Investigación Operativa - Problema 5: Centro de Información Turística  
**Institución:** Universidad Austral de Chile  
**Curso:** Ingeniería Civil en Informática  

---

## Descripción del Problema

Sistema de cola M/M/1 para un centro de información turística donde:
- Llegadas: Proceso Poisson con λ = 5 clientes/min
- Tipos de turistas:
  - **Paciente**: μ = 6 clientes/min (tolera esperas)
  - **Impaciente**: μ = 4 clientes/min (requiere más atención)
- El tipo de turista evoluciona según cadena de Markov con matriz:
  ```
  P = [[0.7, 0.3],
       [0.5, 0.5]]
  ```

---

## Requisitos Técnicos

### Python
- **Versión requerida:** Python 3.8 o superior
- **Recomendado:** Python 3.10-3.12

### Dependencias

| Librería | Versión | Uso |
|----------|---------|-----|
| numpy | 2.3.5 | Cálculos numéricos, variables aleatorias |
| pandas | 2.3.3 | Almacenamiento y análisis de datos |
| matplotlib | 3.10.7 | Generación de gráficos |
| scipy | 1.16.3 | Funciones científicas |
| PyQt5 | 5.15.11 | Interfaz gráfica (opcional) |

**Instalación:**
```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
io-trabajo/
├── main.py                    # Punto de entrada
├── requirements.txt           # Dependencias
├── README.md                  # Este archivo
├── Tarea_Colas_Markov_7.pdf   # Enunciado
├── run_tests.py               # Ejecutor de tests
│
├── src/
│   ├── __init__.py
│   ├── markov.py              # Cadena de Markov
│   ├── queue_theory.py        # Métricas M/M/1
│   ├── simulator.py           # Simulación por eventos discretos
│   └── gui.py                 # Interfaz gráfica (PyQt5)
│
└── tests/
    ├── __init__.py
    ├── test_markov.py
    ├── test_queue_theory.py
    ├── test_simulator.py
    └── test_integration.py
```

---

## Instalación

### Opción 1: Ubuntu/Linux

```bash
# Instalar soporte para entornos virtuales
sudo apt install python3-venv

# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Windows

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 3: macOS

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Instrucciones de Ejecución

### Modo 1: Interfaz Gráfica (GUI)

```bash
python main.py
```

**Funcionalidades:**
- Modificar parámetros de entrada (λ, μ_paciente, μ_impaciente, N)
- Ejecutar simulación con un clic
- Visualizar gráfico N(t) en tiempo real
- Ver tabla comparativa teórico vs. simulado
- Análisis adicional (histogramas, evolución de tipos)

### Modo 2: Línea de Comandos (CLI)

```bash
python main.py --cli
```

**Salida:**
- Distribución estacionaria π
- Métricas analíticas M/M/1
- Resultados de simulación
- Comparación con errores porcentuales
- Gráfico guardado en `figures/n_t_evolution.png`

### Modo 3: Ejecutar Tests

```bash
# Todos los tests
python run_tests.py

# Test individual
python tests/test_markov.py
python tests/test_queue_theory.py
python tests/test_simulator.py
python tests/test_integration.py
```

---

## Parámetros Configurables

Todos los parámetros pueden modificarse directamente en el código o mediante la GUI:

```python
lambda_rate = 5.0           # Tasa de llegadas (clientes/min)
mu_patient = 6.0            # Tasa servicio pacientes (clientes/min)
mu_impatient = 4.0          # Tasa servicio impacientes (clientes/min)
n_tourists = 1000           # Número de turistas a simular

# Matriz de transición Markov
P = [[0.7, 0.3],            # Paciente → [Paciente, Impaciente]
     [0.5, 0.5]]            # Impaciente → [Paciente, Impaciente]
```

**Para cambiar parámetros:**
1. **GUI:** Editar campos en pestaña "Parámetros"
2. **CLI:** Modificar valores en `main.py` (líneas 13-16)
3. **Programático:** Crear instancias de clases con parámetros personalizados

---

## Supuestos del Modelo

### 1. Proceso de Llegadas
- **Supuesto:** Los turistas llegan según un proceso Poisson con tasa λ = 5 clientes/min
- **Implicación:** Tiempos entre llegadas son exponenciales independientes
- **Justificación:** Modelo estándar M/M/1 para llegadas aleatorias

### 2. Tipos de Turistas - Cadena de Markov
- **Supuesto:** El tipo de turista NO es independiente del anterior
- **Modelo:** Cadena de Markov de 2 estados (Paciente/Impaciente)
- **Matriz de transición:** P dada en el enunciado
- **Justificación:** Un turista paciente genera ambiente que atrae pacientes; un impaciente contagia impaciencia

### 3. Tiempos de Servicio
- **Supuesto:** Exponenciales condicionados al tipo de turista
  - Paciente: μ_P = 6 clientes/min
  - Impaciente: μ_I = 4 clientes/min
- **Tasa efectiva:** Se pondera con distribución estacionaria π
- **Cálculo:** μ_eff = 1 / (π_P/μ_P + π_I/μ_I)

### 4. Sistema de Cola
- **Tipo:** M/M/1 (un servidor, cola infinita)
- **Disciplina:** FIFO (First In, First Out)
- **Capacidad:** Ilimitada (no se rechaza a nadie)
- **Abandono:** No hay (nadie se va sin ser atendido)

### 5. Condición Inicial
- **Supuesto:** Sistema comienza vacío: N(0) = 0
- **Implicación:** Simulación captura régimen transitorio
- **Consecuencia:** Métricas simuladas difieren de teoría estacionaria para N pequeños

### 6. Distribución Estacionaria π
- **Cálculo:** Resolviendo π·P = π y Σπ_i = 1
- **Resultado:** π = [0.625, 0.375]
- **Uso:** Ponderar tiempos de servicio para calcular μ_eff

### 7. Estabilidad del Sistema
- **Condición:** ρ = λ/μ_eff < 1
- **Con parámetros dados:** ρ ≈ 0.9896 < 1 (estable, pero muy saturado)
- **Nota:** Alta utilización (99%) causa convergencia lenta al estado estacionario

### 8. Simulación por Eventos Discretos
- **Eventos:** Llegadas (+1 cliente) y Salidas (-1 cliente)
- **N(t):** Calculado mediante método del área (integración numérica)
- **Métricas:** Ley de Little (L = λ·W) y método del área
- **Reproducibilidad:** Semillas aleatorias permiten replicar resultados

---

## Outputs del Simulador

### Console Output (CLI)
```
Distribución estacionaria π:
  Paciente: 0.6250
  Impaciente: 0.3750

Métricas M/M/1 ponderadas:
  rho: 0.9896
  L: 95.00
  W: 19.00
  Lq: 94.01
  Wq: 18.80

Métricas simuladas:
  L (clientes en sistema): 15.73
  W (tiempo en sistema): 3.17 min
  ...
```

### DataFrame Resultados
Columnas incluidas en `results`:

| Columna | Descripción |
|---------|-------------|
| turista | ID del turista (1, 2, 3, ...) |
| tiempo_llegada | Tiempo de llegada al sistema |
| tiempo_inicio_servicio | Cuándo inicia el servicio |
| tiempo_salida | Cuándo sale del sistema |
| tipo | "Paciente" o "Impaciente" |
| estado | 0 (Paciente) o 1 (Impaciente) |
| tiempo_servicio | Duración del servicio |
| tiempo_espera | Tiempo en cola |
| tiempo_en_sistema | Tiempo total (espera + servicio) |
| n_sistema | N(t) al momento de llegar |

**Atributos adicionales** (`results.attrs`):
- `L_simulated`: Clientes promedio en sistema
- `W_simulated`: Tiempo promedio en sistema
- `Lq_simulated`: Clientes promedio en cola
- `Wq_simulated`: Tiempo promedio en cola
- `N_steady_state`: N(t) en últimos 20%
- `lambda_empirical`: Tasa de llegadas empírica
- `total_time`: Tiempo total de simulación

### Gráficos Generados
- **CLI:** `figures/n_t_evolution.png` (evolución de N(t))
- **GUI:** Visualización interactiva en ventana

---

## Validación del Código

El código incluye tests automatizados que verifican:

### Tests de Markov (`test_markov.py`)
- ✅ π suma 1
- ✅ π·P = π (propiedad estacionaria)
- ✅ Generación correcta de estados

### Tests de Teoría de Colas (`test_queue_theory.py`)
- ✅ Cálculo de μ_eff ponderada
- ✅ Métricas M/M/1 (L, W, Lq, Wq)
- ✅ Detección de inestabilidad (ρ ≥ 1)
- ✅ Ley de Little

### Tests del Simulador (`test_simulator.py`)
- ✅ Estructura del DataFrame
- ✅ Distribución de tipos coincide con π
- ✅ Disciplina FIFO respetada
- ✅ Consistencia interna (Ley de Little)

### Tests de Integración (`test_integration.py`)
- ✅ Sistema completo funciona correctamente
- ✅ Reproducibilidad con semillas
- ✅ Diferentes escenarios de carga

**Ejecutar todos los tests:**
```bash
python run_tests.py
# Resultado esperado: 4/4 tests pasaron
```

---

## Ejemplo de Uso Programático

```python
from src.markov import MarkovChain
from src.queue_theory import MMOneQueue
from src.simulator import TouristCenterSimulator

# Parámetros
lambda_rate = 5.0
mu_patient = 6.0
mu_impatient = 4.0
P = [[0.7, 0.3], [0.5, 0.5]]

# 1. Cadena de Markov
markov = MarkovChain(P)
pi = markov.get_stationary_distribution()
print(f"π = {pi}")  # {'Paciente': 0.625, 'Impaciente': 0.375}

# 2. Métricas analíticas
queue = MMOneQueue(lambda_rate, mu_patient, mu_impatient, markov.pi)
metrics = queue.calculate_metrics()
print(f"ρ = {metrics['rho']:.4f}")  # 0.9896
print(f"L = {metrics['L']:.2f}")     # 95.00

# 3. Simulación
simulator = TouristCenterSimulator(lambda_rate, mu_patient, mu_impatient, markov)
results = simulator.simulate(1000)

# 4. Resultados
print(f"L simulado = {results.attrs['L_simulated']:.2f}")  # 15.73
print(f"Proporción pacientes = {(results['tipo']=='Paciente').mean():.4f}")  # 0.6260
```

---

## Notas Importantes

### 1. Alta Utilización (ρ ≈ 0.99)
- El sistema opera al 99% de capacidad
- Causa alta variabilidad en resultados
- Convergencia lenta al estado estacionario
- Diferencias entre teoría y simulación son esperadas para N=1,000

### 2. Régimen Transitorio
- Simulaciones de N=1,000 capturan fase inicial (transitoria)
- Para alcanzar estado estacionario se requieren N>50,000
- Métricas simuladas (L≈15) representan primeras horas de operación
- Métricas teóricas (L≈95) representan comportamiento a largo plazo

### 3. Reproducibilidad
- Tests usan semillas aleatorias fijas
- Para replicar resultados exactos, usar misma semilla
- Ejemplo: `np.random.seed(42)` antes de simular

---

## Solución de Problemas

### Error: ModuleNotFoundError
```bash
# Asegurarse de estar en el entorno virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Reinstalar dependencias
pip install -r requirements.txt
```

### Error: No se puede importar PyQt5 (GUI)
```bash
# Instalar PyQt5 manualmente
pip install PyQt5

# Alternativa: usar solo modo CLI
python main.py --cli
```

### Gráficos no se muestran
```bash
# Verificar backend de matplotlib
python -c "import matplotlib; print(matplotlib.get_backend())"

# Si es necesario, cambiar backend
export MPLBACKEND=TkAgg  # Linux/Mac
set MPLBACKEND=TkAgg     # Windows
```

---

## Autores

- Bastián Gajardo
- Benjamín Martínez
- Cristóbal Skillmann
- Katherine Zapata

**Profesor:** Tania Letelier  
**Institución:** Universidad Austral de Chile  
**Curso:** Investigación Operativa  
**Fecha:** 24 Noviembre 2025

---

## Licencia

Apache License 2.0 - Ver archivo `LICENSE` para detalles.

---

## Referencias

- Hillier & Lieberman — *Introducción a la Investigación de Operaciones*
- Gross & Harris — *Fundamentals of Queueing Theory*
- Norris — *Markov Chains*
