# **Simulador M/M/1 con Cadena de Markov — Investigación Operativa**

Este proyecto implementa un **modelo de línea de espera (cola) M/M/1** donde el tiempo de servicio depende del **tipo de turista**, el cual evoluciona según una **cadena de Markov de dos estados**.
El objetivo es analizar el comportamiento del sistema, comparar la teoría con la simulación y estudiar estabilidad, tiempos de espera, distribución estacionaria y desempeño.

Este trabajo integra conceptos centrales del ramo de **Investigación Operativa**:

* **Líneas de espera (M/M/1)**
* **Cadenas de Markov**
* **Simulación de eventos discretos**
* **Evaluación de estabilidad**
* **Comparación entre teoría y datos simulados**


# **Estructura del Proyecto**

io-trabajo/
├── main.py                    # Punto de entrada del programa
├── requirements.txt           # Dependencias del proyecto
├── Tarea_Colas_Markov_7.pdf   # Documento teórico o enunciado
├── LICENSE
├── .gitignore
│
├── src/
│   ├── **init**.py
│   ├── gui.py                 # Interfaz gráfica con Tkinter
│   ├── markov.py              # Funciones y modelos de cadenas de Markov
│   ├── queue_theory.py        # Modelos analíticos de colas M/M/1, M/M/s, etc.
│   └── simulator.py           # Simulación por eventos discretos
│
├── tests/
│   └── **init**.py            # Carpeta preparada para tests automáticos
│
└── .venv/                     # Entorno virtual (no se sube al repo)

# **Dependencias**

**Python recomendado: 3.10–3.12**

Librerías usadas:

* numpy
* pandas
* matplotlib
* scipy
* PyQt5
* pillow
* tzdata

Instalación automática:

```bash
pip install -r requirements.txt
```

# **Instalación del Entorno (Ubuntu, PEP 668)**

Ubuntu bloquea `pip` global → **NECESARIO usar venv**.

```bash
sudo apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


# **Ejecución del Proyecto**

## **1. Modo GUI (recomendado)**

```bash
python3 main.py
```

La interfaz permite:

* Ingresar parámetros (λ, μ_p, μ_i, P y número de turistas)
* Ejecutar la simulación sin bloquear la ventana (uso de QThread)
* Ver:

  * Gráfico N(t)
  * Tabla métrica teórica vs simulada
  * Consola explicativa


## **2. Modo CLI**

```bash
python3 main.py --cli
```

Muestra en consola:

* Distribución estacionaria π
* Métricas analíticas M/M/1
* Métricas simuladas y comparación
* Estadísticas adicionales
* Genera gráfico `figures/n_t_evolution.png`


# **Descripción IO del Modelo**

Este trabajo implementa un **sistema M/M/1**, donde:

* **Llegadas:** Proceso Poisson con tasa λ
* **Servicio:** Exponencial con dos tasas (μ_paciente, μ_impaciente)
* **Tipo de turista:** Determinado por una **cadena de Markov**, cumpliendo:
  [
  P = \begin{bmatrix}
  0.7 & 0.3 \
  0.5 & 0.5
  \end{bmatrix}
  ]

Esto genera un modelo híbrido:

> **Llegadas Poisson + mezcla Markoviana de tiempos de servicio + un servidor FIFO**

La simulación utiliza **eventos discretos**, cálculo de tiempos de llegada, inicio de servicio, salida y N(t).


# **📊 Métricas Analíticas M/M/1**

El sistema calcula:

[
\mu_{\text{eff}} = \frac{1}{\pi_p(1/\mu_p) + \pi_i(1/\mu_i)}
]

[
\rho = \frac{\lambda}{\mu_{\text{eff}}}
]

[
L = \frac{\rho}{1 - \rho},\quad
L_q = \frac{\rho^2}{1 - \rho}
]

[
W = \frac{1}{\mu_{\text{eff}} - \lambda}, \quad
W_q = \frac{\rho}{\mu_{\text{eff}} - \lambda}
]


# **Supuestos del Modelo**

## **1. Proceso de llegadas**

* Los turistas llegan siguiendo un **proceso Poisson**.
* Los tiempos entre llegadas son **exponenciales**.

Este es un supuesto estándar del modelo **M/M/1**.

## **2. Cadena de Markov para los tipos**

* Los tipos de turistas **no son independientes**.
* Se modela una cadena de Markov con dos estados:

  * Paciente
  * Impaciente
* La transición del tipo del turista **sigue la matriz P** entregada.

Esto conecta directamente con el capítulo de **Cadenas de Markov** de IO.

## **3. Tiempos de servicio**

* Exponencial condicionado al tipo:

  * Paciente → μ_p = 6
  * Impaciente → μ_i = 4
* La tasa efectiva se pondera con π.

## **4. Sistema de cola M/M/1**

* Un servidor.
* Disciplina FIFO.
* Cola de capacidad infinita.
* Nadie abandona.

Modelo clásico **M/M/1** visto en líneas de espera.

## **5. Estabilidad**

El sistema es **estable si**:

[
\rho=\frac{\lambda}{\mu_{\text{eff}}} < 1
]

La simulación evalúa:

* **ρ**
* **N(t) promedio**
* **N(t) al final (20% último)** → aproximación empírica a estado estacionario

Esto responde a la parte de teoría y simulación.

## **6. Simulación de eventos discretos**

* Llegadas generadas previamente
* Tiempos de servicio generados según tipo
* Se procesan eventos:

  * Llegada (+1)
  * Salida (-1)
* Se registra N(t) usando el método del área.

Técnica estándar del capítulo de **Simulación en IO**.

## **7. Comparación teoría–simulación**

El sistema entrega:

* L, W, Lq, Wq **teóricos**
* L_sim, W_sim, etc. **simulados**
* Error porcentual entre ambos

Este análisis es parte core del enfoque IO.

# **Resultados Guardados en la Simulación**

El DataFrame final incluye:

| Columna                | Descripción                        |
| ---------------------- | ---------------------------------- |
| turista                | ID del turista                     |
| tiempo_llegada         | Tiempo absoluto de llegada         |
| tiempo_inicio_servicio | Momento en que inicia servicio     |
| tiempo_salida          | Momento en que abandona el sistema |
| tipo                   | Paciente/Impaciente                |
| estado                 | 0 o 1 según cadena de Markov       |
| tiempo_servicio        | Duración del servicio              |
| tiempo_espera          | Tiempo en cola                     |
| tiempo_en_sistema      | Tiempo total                       |
| n_sistema              | N(t) en cada llegada               |

Atributos adicionales:

* L_sim, W_sim, Lq_sim, Wq_sim
* N_steady_state (último 20%)
* lambda_empirical
* total_time

# **Tests**

La carpeta `tests/` mantiene estructura base.
La simulación fue verificada con λ=5, μ_p=6, μ_i=4 y matriz P del enunciado.


# **Relación con los contenidos del ramo**

| Contenido IO            | Cómo se aborda                         |
| ----------------------- | -------------------------------------- |
| Cadenas de Markov       | Matriz P, π, evolución del tipo        |
| Líneas de espera        | Modelo M/M/1 con parámetros efectivos  |
| Simulación              | Eventos discretos, tiempos, N(t)       |
| Estabilidad             | ρ, estado estacionario empírico        |
| Modelos probabilísticos | Distribuciones exponenciales y Poisson |


# **Referencias IO relevantes**

* Hillier & Lieberman — *Introducción a la Investigación de Operaciones*
* Gross & Harris — *Fundamentals of Queueing Theory*
* Winston — *Operations Research*
