# **Simulador M/M/1 con Cadena de Markov — Investigación Operativa**

## **Descripción del Proyecto**

Este proyecto implementa un **modelo de línea de espera (cola) M/M/1** donde el tiempo de servicio depende del **tipo de turista**, el cual evoluciona según una **cadena de Markov de dos estados**.

El sistema modela un **Centro de Información Turística** que recibe turistas en busca de orientación. Cada turista puede ser:
- **Paciente**: tolera esperas largas y coopera durante la atención (μ = 6 clientes/min)
- **Impaciente**: se irrita fácilmente y necesita más tiempo de atención (μ = 4 clientes/min)

El tipo de turista no es completamente aleatorio, sino que depende del tipo anterior según una cadena de Markov discreta con matriz de transición:

```
P = [[0.7, 0.3],    # Paciente → [Paciente, Impaciente]
     [0.5, 0.5]]    # Impaciente → [Paciente, Impaciente]
```

---

## **Objetivos de la Tarea**

Este trabajo integra conceptos centrales del ramo de **Investigación Operativa**:

### **Tareas Implementadas**

✅ **a) Calcular π estacionaria de la cadena de Markov**
- Implementado en `src/markov.py`
- Método: `MarkovChain.get_stationary_distribution()`
- Cálculo mediante valores propios de P^T

✅ **b) Implementar cálculo ponderado de ρ, L y W**
- Implementado en `src/queue_theory.py`
- Clase: `MMOneQueue`
- Calcula métricas analíticas usando distribución estacionaria π

✅ **c) Simulador Python para 1000 periodos**
- Implementado en `src/simulator.py`
- Clase: `TouristCenterSimulator`
- Registra:
  - N(t): número total de turistas en el sistema
  - Tipo de turista actual (Paciente/Impaciente)
  - Estado del sistema (estable/inestable)

✅ **d) Conclusiones basadas en simulaciones repetidas**
- Incluidas en el informe PDF
- Análisis de convergencia, variabilidad y comportamiento estacionario

---

## **Estructura del Proyecto**

```
io-trabajo/
├── main.py                    # Punto de entrada del programa
├── requirements.txt           # Dependencias del proyecto
├── Tarea_Colas_Markov_7.pdf   # Enunciado de la tarea
├── README.md                  # Este archivo
├── LICENSE                    # Licencia Apache 2.0
├── .gitignore                 # Archivos a ignorar
├── run_tests.py               # Script para ejecutar todos los tests
│
├── src/                       # Código fuente
│   ├── __init__.py
│   ├── gui.py                 # Interfaz gráfica con PyQt5
│   ├── markov.py              # Cadenas de Markov (π estacionaria)
│   ├── queue_theory.py        # Modelos analíticos M/M/1
│   └── simulator.py           # Simulación por eventos discretos
│
└── tests/                     # Tests unitarios
    ├── __init__.py
    ├── test_markov.py         # Tests para cadena de Markov
    ├── test_queue_theory.py   # Tests para teoría de colas
    ├── test_simulator.py      # Tests para simulador
    └── test_integration.py    # Tests de integración completa
```

---

## **Requisitos Técnicos**

### **Versión de Python**
- **Python 3.8+** (recomendado: 3.10-3.12)

### **Dependencias**

El proyecto utiliza las siguientes librerías:

| Librería | Versión | Propósito |
|----------|---------|-----------|
| `numpy` | 2.3.5 | Cálculos numéricos, generación de variables aleatorias |
| `pandas` | 2.3.3 | Manipulación de datos, almacenamiento de resultados |
| `matplotlib` | 3.10.7 | Generación de gráficos y visualizaciones |
| `scipy` | 1.16.3 | Funciones científicas avanzadas |
| `PyQt5` | 5.15.11 | Interfaz gráfica de usuario |
| `pillow` | 12.0.0 | Procesamiento de imágenes para GUI |

**Instalación automática:**

```bash
pip install -r requirements.txt
```

---

## **Instalación del Entorno**

### **En Ubuntu (PEP 668)**

Ubuntu bloquea `pip` global por seguridad, por lo que es **NECESARIO usar un entorno virtual**:

```bash
# Instalar soporte para entornos virtuales
sudo apt install python3.12-venv

# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### **En Windows**

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### **En macOS**

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## **Ejecución del Proyecto**

### **1. Modo GUI (Recomendado)**

```bash
python3 main.py
```

La interfaz gráfica permite:
- ✨ Ingresar parámetros (λ, μ_paciente, μ_impaciente, número de turistas)
- 🚀 Ejecutar simulación sin bloquear la ventana (QThread)
- 📊 Ver gráfico N(t) en tiempo real
- 📈 Análisis adicional (histogramas, evolución de tipos, boxplots)
- 📋 Tabla comparativa de métricas teóricas vs simuladas
- 🖥️ Consola con resultados detallados

**Interfaz incluye:**
- **Tab 1**: Configuración de parámetros
- **Tab 2**: Gráfico N(t) principal
- **Tab 3**: Análisis adicional (4 subgráficos)
- **Tab 4**: Tabla de métricas
- **Tab 5**: Resultados en formato texto

### **2. Modo CLI (Línea de Comandos)**

```bash
python3 main.py --cli
```

Muestra en consola:
- Distribución estacionaria π
- Métricas analíticas M/M/1 (ρ, L, W, Lq, Wq)
- Métricas simuladas y comparación
- Estadísticas adicionales
- Genera gráfico `figures/n_t_evolution.png`

### **3. Ejecutar Tests**

```bash
# Todos los tests
python3 run_tests.py

# Test individual
python3 tests/test_markov.py
python3 tests/test_queue_theory.py
python3 tests/test_simulator.py
python3 tests/test_integration.py
```

---

## **Parámetros Configurables**

El código permite cambiar todos los parámetros de entrada:

```python
# En main.py o mediante GUI
lambda_rate = 5.0           # Tasa de llegadas (clientes/min)
mu_patient = 6.0            # Tasa servicio pacientes (clientes/min)
mu_impatient = 4.0          # Tasa servicio impacientes (clientes/min)
P = [[0.7, 0.3],           # Matriz de transición Markov
     [0.5, 0.5]]
n_tourists = 1000          # Número de turistas a simular
```

**Desde GUI**: Todos los parámetros son editables en la pestaña "Parámetros"

---

## **Modelo Matemático**

### **1. Cadena de Markov**

La distribución estacionaria π satisface:

```
π · P = π
π₁ + π₂ = 1
```

Solución para P = [[0.7, 0.3], [0.5, 0.5]]:
```
π = [0.625, 0.375]
```

### **2. Tiempo de Servicio Efectivo**

```
E[S] = π_p · (1/μ_p) + π_i · (1/μ_i)
μ_eff = 1 / E[S]
```

Con los parámetros del problema:
```
E[S] = 0.625·(1/6) + 0.375·(1/4) = 0.1979 min
μ_eff = 5.052 clientes/min
```

### **3. Métricas M/M/1**

```
ρ = λ / μ_eff                    # Utilización del servidor
L = ρ / (1 - ρ)                  # Clientes en sistema
W = 1 / (μ_eff - λ)             # Tiempo en sistema
Lq = ρ² / (1 - ρ)               # Clientes en cola
Wq = ρ / (μ_eff - λ)            # Tiempo en cola
```

### **4. Ley de Little**

```
L = λ · W
Lq = λ · Wq
```

---

## **Supuestos del Modelo**

### **1. Proceso de Llegadas**
- Los turistas llegan según un **proceso Poisson** con tasa λ = 5 clientes/min
- Los tiempos entre llegadas son **exponenciales** con media 1/λ

### **2. Cadena de Markov**
- Los tipos de turistas **no son independientes**
- Se modela con cadena de Markov de 2 estados (Paciente/Impaciente)
- La transición sigue la matriz P dada

### **3. Tiempos de Servicio**
- **Exponencial** condicionado al tipo de turista
- μ_paciente = 6 clientes/min
- μ_impaciente = 4 clientes/min
- La tasa efectiva se pondera con π

### **4. Sistema M/M/1**
- **Un solo servidor**
- Disciplina **FIFO** (First In, First Out)
- **Cola de capacidad infinita**
- **Nadie abandona** el sistema

### **5. Estabilidad**
El sistema es estable si:
```
ρ = λ / μ_eff < 1
```

Con los parámetros del problema:
```
ρ = 5 / 5.052 ≈ 0.99 (estable, pero muy cercano al límite)
```

### **6. Simulación**
- **Eventos discretos**: llegadas y salidas
- Tiempos de llegada generados con `np.random.exponential(1/λ)`
- Tiempos de servicio según tipo de turista
- N(t) calculado mediante método del área (integración numérica)

---

## **Resultados de la Simulación**

El DataFrame final incluye las siguientes columnas:

| Columna | Descripción |
|---------|-------------|
| `turista` | ID del turista (1, 2, 3, ...) |
| `tiempo_llegada` | Tiempo absoluto de llegada al sistema |
| `tiempo_inicio_servicio` | Momento en que inicia el servicio |
| `tiempo_salida` | Momento en que abandona el sistema |
| `tipo` | "Paciente" o "Impaciente" |
| `estado` | 0 (Paciente) o 1 (Impaciente) |
| `tiempo_servicio` | Duración del servicio |
| `tiempo_espera` | Tiempo en cola esperando |
| `tiempo_en_sistema` | Tiempo total (espera + servicio) |
| `n_sistema` | N(t): turistas en sistema al llegar |

**Atributos adicionales** (guardados en `results.attrs`):
- `L_simulated`: Clientes promedio en sistema
- `W_simulated`: Tiempo promedio en sistema
- `Lq_simulated`: Clientes promedio en cola
- `Wq_simulated`: Tiempo promedio en cola
- `N_steady_state`: N(t) en estado estacionario (últimos 20%)
- `lambda_empirical`: Tasa de llegadas empírica
- `total_time`: Tiempo total de simulación

---

## **Tests Implementados**

### **Tests Unitarios**

1. **test_markov.py**: Verifica cadena de Markov
   - Distribución estacionaria π
   - Propiedad π·P = π
   - Generación de estados

2. **test_queue_theory.py**: Verifica métricas analíticas
   - Cálculo de μ efectiva
   - Métricas M/M/1 (L, W, Lq, Wq)
   - Detección de inestabilidad (ρ ≥ 1)
   - Ley de Little

3. **test_simulator.py**: Verifica simulación
   - Estructura del DataFrame
   - Distribución de tipos según π
   - Disciplina FIFO
   - Cálculo de N(t)
   - Consistencia de métricas

4. **test_integration.py**: Tests de integración
   - Sistema completo con parámetros del problema
   - Reproducibilidad con semillas fijas
   - Diferentes escenarios de carga
   - Estado estacionario

### **Ejecutar Tests**

```bash
# Todos los tests
python3 run_tests.py

# Resultado esperado:
# [PASS] Test 1: Cadena de Markov
# [PASS] Test 2: Teoría de Colas M/M/1
# [PASS] Test 3: Simulador
# [PASS] Test 4: Integración Completa
# Total: 4/4 tests pasaron
```

---

## **Entregables**

### ✅ **1. Código Python** (`src/*.py`)
- **Comentado**: Todas las funciones tienen docstrings
- **Modular**: Separado en módulos independientes
- **Parametrizable**: Todos los parámetros son configurables
- **Testeado**: 4 archivos de tests con >20 casos de prueba

### ✅ **2. Informe en PDF**
Contiene:
- **Portada**: Título, integrantes, fecha
- **Índice**: Estructura del documento
- **Introducción**: Descripción del problema
- **Resultados**: Métricas teóricas y simuladas
- **Gráficos**: N(t), histogramas, evolución de tipos
- **Conclusiones**: Análisis del comportamiento del sistema

### ✅ **3. README.md** (este archivo)
Incluye:
- Instrucciones de instalación
- Cómo ejecutar el código (GUI y CLI)
- Documentación de parámetros
- Descripción de supuestos
- Explicación del modelo matemático

---

## **Relación con Investigación Operativa**

| Contenido IO | Implementación en el Proyecto |
|-------------|-------------------------------|
| **Cadenas de Markov** | Matriz P, cálculo de π, evolución del tipo de turista |
| **Líneas de Espera** | Modelo M/M/1 con parámetros efectivos ponderados |
| **Simulación** | Eventos discretos, generación de tiempos, N(t) |
| **Estabilidad** | Análisis de ρ, estado estacionario empírico |
| **Ley de Little** | Verificación de L = λ·W y Lq = λ·Wq |
| **Modelos Probabilísticos** | Distribuciones exponenciales y Poisson |

---

## **Ejemplo de Uso**

### **Ejecutar simulación desde código**

```python
from src.markov import MarkovChain
from src.queue_theory import MMOneQueue
from src.simulator import TouristCenterSimulator

# Parámetros
lambda_rate = 5.0
mu_patient = 6.0
mu_impatient = 4.0
P = [[0.7, 0.3], [0.5, 0.5]]

# Cadena de Markov
markov = MarkovChain(P)
pi = markov.get_stationary_distribution()
print(f"π = {pi}")

# Métricas analíticas
queue = MMOneQueue(lambda_rate, mu_patient, mu_impatient, markov.pi)
metrics = queue.calculate_metrics()
print(f"ρ = {metrics['rho']:.4f}")
print(f"L = {metrics['L']:.2f}")

# Simulación
simulator = TouristCenterSimulator(lambda_rate, mu_patient, mu_impatient, markov)
results = simulator.simulate(1000)
print(f"L simulado = {results.attrs['L_simulated']:.2f}")
```

---

## **Notas Importantes**

### **⚠️ Alta Utilización (ρ ≈ 0.99)**

Con los parámetros del problema:
- ρ = 5 / 5.052 ≈ **0.99** (muy cerca del límite de estabilidad)
- Esto genera **alta variabilidad** en las métricas
- Se requieren simulaciones **largas** (>5000 turistas) para buena convergencia
- Diferencias del **10-20%** entre teoría y simulación son esperables

### **📊 Convergencia**

Para mejor convergencia:
- Usar `n_tourists >= 5000`
- Analizar solo el estado estacionario (últimos 20-30%)
- Ejecutar múltiples réplicas y promediar

### **🔬 Validación**

El sistema ha sido validado con:
- ✅ Tests unitarios automatizados
- ✅ Verificación de Ley de Little
- ✅ Comparación con teoría M/M/1
- ✅ Distribución de tipos coincide con π

---

## **Referencias**

- Hillier & Lieberman — *Introducción a la Investigación de Operaciones*
- Gross & Harris — *Fundamentals of Queueing Theory*
- Winston — *Operations Research*
- Norris — *Markov Chains*

---

## **Licencia**

Este proyecto está bajo la licencia Apache 2.0. Ver archivo `LICENSE` para más detalles.

---

## **Autores**

Trabajo realizado para el curso de **Investigación Operativa**.

-Bastian Gajardo
-Benjamin Martinez
-Cristobal Skillmann
-Katherine Zapata

**Fecha**: 24/11/2025

---
