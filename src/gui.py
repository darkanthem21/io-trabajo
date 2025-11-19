import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.markov import MarkovChain
from src.queue_theory import MMOneQueue
from src.simulator import TouristCenterSimulator


class SimulationThread(QThread):
    """Thread para no bloquear UI"""

    finished = pyqtSignal(pd.DataFrame, dict, dict)

    def __init__(self, lambda_rate, mu_p, mu_i, P, n_periods):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.mu_p = mu_p
        self.mu_i = mu_i
        self.P = P
        self.n_periods = n_periods

    def run(self):
        markov = MarkovChain(self.P)
        pi_dist = markov.get_stationary_distribution()

        queue = MMOneQueue(self.lambda_rate, self.mu_p, self.mu_i, markov.pi)
        metrics = queue.calculate_metrics()

        simulator = TouristCenterSimulator(
            self.lambda_rate, self.mu_p, self.mu_i, markov
        )
        results = simulator.simulate(self.n_periods)

        self.finished.emit(results, pi_dist, metrics)


class MplCanvas(FigureCanvasQTAgg):
    """Canvas de Matplotlib"""

    def __init__(self, parent=None, width=10, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador M/M/1 + Cadena de Markov")
        self.setGeometry(100, 100, 1200, 800)

        # Valores por defecto
        self.lambda_rate = 5.0
        self.mu_p = 6.0
        self.mu_i = 4.0
        self.n_periods = 1000
        self.P = [[0.7, 0.3], [0.5, 0.5]]

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Parámetros
        tab1 = self.create_params_tab()
        tabs.addTab(tab1, "Parámetros")

        # Tab 2: Gráficos
        self.canvas = MplCanvas(self, width=10, height=6)
        tabs.addTab(self.canvas, "Gráfico N(t)")

        # Tab 3: Métricas
        self.metrics_table = QTableWidget()
        tabs.addTab(self.metrics_table, "Métricas")

        # Tab 4: Resultados
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        tabs.addTab(self.results_text, "Consola")

    def create_params_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Grupo parámetros
        group = QGroupBox("Configuración del Sistema")
        form = QFormLayout()

        self.lambda_input = QLineEdit(str(self.lambda_rate))
        self.mu_p_input = QLineEdit(str(self.mu_p))
        self.mu_i_input = QLineEdit(str(self.mu_i))
        self.periods_input = QLineEdit(str(self.n_periods))

        form.addRow("λ (llegadas/min):", self.lambda_input)
        form.addRow("μ Paciente (servicio/min):", self.mu_p_input)
        form.addRow("μ Impaciente (servicio/min):", self.mu_i_input)
        form.addRow("Número de turistas:", self.periods_input)

        group.setLayout(form)
        layout.addWidget(group)

        # Botón simular
        self.sim_button = QPushButton("Ejecutar Simulación")
        self.sim_button.setStyleSheet("font-size: 14pt; padding: 10px;")
        self.sim_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.sim_button)

        # Status
        self.status_label = QLabel("Listo para simular")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addStretch()
        return widget

    def run_simulation(self):
        # Leer parámetros
        try:
            self.lambda_rate = float(self.lambda_input.text())
            self.mu_p = float(self.mu_p_input.text())
            self.mu_i = float(self.mu_i_input.text())
            self.n_periods = int(self.periods_input.text())
        except ValueError:
            self.status_label.setText("❌ Error: parámetros inválidos")
            return

        # Deshabilitar botón
        self.sim_button.setEnabled(False)
        self.status_label.setText("Simulando...")

        # Ejecutar en thread
        self.thread = SimulationThread(
            self.lambda_rate, self.mu_p, self.mu_i, self.P, self.n_periods
        )
        self.thread.finished.connect(self.on_simulation_finished)
        self.thread.start()

    def on_simulation_finished(self, results, pi_dist, metrics):
        # Gráfico
        self.canvas.axes.clear()
        self.canvas.axes.plot(results["turista"], results["n_sistema"])
        self.canvas.axes.set_xlabel("Turista")
        self.canvas.axes.set_ylabel("N(t) - Turistas en sistema")
        self.canvas.axes.set_title("Evolución del Sistema")
        self.canvas.axes.grid(True)
        self.canvas.draw()

        # Métricas
        self.update_metrics_table(pi_dist, metrics, results)

        # Obtener métricas simuladas
        L_sim = results.attrs.get("L_simulated", 0)
        W_sim = results.attrs.get("W_simulated", 0)
        Lq_sim = results.attrs.get("Lq_simulated", 0)
        Wq_sim = results.attrs.get("Wq_simulated", 0)

        # Calcular errores porcentuales
        error_L = (
            abs(metrics["L"] - L_sim) / metrics["L"] * 100 if metrics["L"] > 0 else 0
        )
        error_W = (
            abs(metrics["W"] - W_sim) / metrics["W"] * 100 if metrics["W"] > 0 else 0
        )
        error_Lq = (
            abs(metrics["Lq"] - Lq_sim) / metrics["Lq"] * 100
            if metrics["Lq"] > 0
            else 0
        )
        error_Wq = (
            abs(metrics["Wq"] - Wq_sim) / metrics["Wq"] * 100
            if metrics["Wq"] > 0
            else 0
        )

        # Consola
        output = f"""
Distribución estacionaria π:
  Paciente: {pi_dist["Paciente"]:.4f}
  Impaciente: {pi_dist["Impaciente"]:.4f}

Métricas M/M/1 teóricas:
  ρ (utilización): {metrics["rho"]:.4f}
  L (clientes en sistema): {metrics["L"]:.2f}
  W (tiempo en sistema): {metrics["W"]:.2f} min
  Lq (clientes en cola): {metrics["Lq"]:.2f}
  Wq (tiempo en cola): {metrics["Wq"]:.2f} min

Métricas simuladas:
  L (clientes en sistema): {L_sim:.2f}
  W (tiempo en sistema): {W_sim:.2f} min
  Lq (clientes en cola): {Lq_sim:.2f}
  Wq (tiempo en cola): {Wq_sim:.2f} min

Comparación (error %):
  L:  {error_L:.1f}%
  W:  {error_W:.1f}%
  Lq: {error_Lq:.1f}%
  Wq: {error_Wq:.1f}%

Estadísticas adicionales:
  N(t) promedio en llegadas: {results["n_sistema"].mean():.2f}
  N(t) estado estacionario: {results.attrs.get("N_steady_state", 0):.2f}

Distribución tipos:
{results["tipo"].value_counts(normalize=True)}
        """
        self.results_text.setText(output)

        # Reactivar
        self.sim_button.setEnabled(True)
        self.status_label.setText("Simulación completada")

    def update_metrics_table(self, pi_dist, metrics, results):
        # Obtener métricas simuladas
        L_sim = results.attrs.get("L_simulated", 0)
        W_sim = results.attrs.get("W_simulated", 0)
        Lq_sim = results.attrs.get("Lq_simulated", 0)
        Wq_sim = results.attrs.get("Wq_simulated", 0)

        data = [
            ("π Paciente", f"{pi_dist['Paciente']:.4f}", ""),
            ("π Impaciente", f"{pi_dist['Impaciente']:.4f}", ""),
            ("", "", ""),  # Separador
            ("ρ (utilización)", f"{metrics['rho']:.4f}", ""),
            ("", "", ""),  # Separador
            ("L teórico", f"{metrics['L']:.2f}", "clientes"),
            ("L simulado", f"{L_sim:.2f}", "clientes"),
            ("", "", ""),  # Separador
            ("W teórico", f"{metrics['W']:.2f}", "min"),
            ("W simulado", f"{W_sim:.2f}", "min"),
            ("", "", ""),  # Separador
            ("Lq teórico", f"{metrics['Lq']:.2f}", "clientes"),
            ("Lq simulado", f"{Lq_sim:.2f}", "clientes"),
            ("", "", ""),  # Separador
            ("Wq teórico", f"{metrics['Wq']:.2f}", "min"),
            ("Wq simulado", f"{Wq_sim:.2f}", "min"),
        ]

        self.metrics_table.setRowCount(len(data))
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor", "Unidad"])

        for i, row in enumerate(data):
            for j, value in enumerate(row):
                self.metrics_table.setItem(i, j, QTableWidgetItem(value))

        self.metrics_table.resizeColumnsToContents()


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
