import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
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


class MultiplotCanvas(FigureCanvasQTAgg):
    """Canvas con múltiples subplots"""

    def __init__(self, parent=None, nrows=2, ncols=2, width=12, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig = fig
        self.axes = fig.subplots(nrows, ncols)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador M/M/1 + Cadena de Markov")
        self.setGeometry(50, 50, 1400, 900)

        # Valores por defecto
        self.lambda_rate = 5.0
        self.mu_p = 6.0
        self.mu_i = 4.0
        self.n_periods = 1000
        self.P = [[0.7, 0.3], [0.5, 0.5]]

        # Almacenar resultados de la última simulación
        self.last_results = None
        self.last_pi_dist = None
        self.last_metrics = None

        self.setup_ui()
        self.apply_styles()

    def apply_styles(self):
        """Aplica estilos minimalistas"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f5f5f5;
                color: #333333;
                padding: 8px 16px;
                border: 1px solid #d0d0d0;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QGroupBox {
                border: 1px solid #d0d0d0;
                margin-top: 8px;
                padding-top: 8px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 5px;
                color: #333333;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #d0d0d0;
                background-color: white;
            }
            QLineEdit:focus {
                border: 1px solid #666666;
            }
            QPushButton {
                background-color: #333333;
                color: white;
                border: none;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                color: #333333;
            }
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: white;
                border: 1px solid #d0d0d0;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                color: #333333;
                padding: 6px;
                border: 1px solid #d0d0d0;
            }
            QTextEdit {
                font-family: 'Courier New', monospace;
                background-color: white;
                border: 1px solid #d0d0d0;
            }
        """)

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

        # Tab 2: Gráfico principal
        tab2 = self.create_graph_tab()
        tabs.addTab(tab2, "Gráfico N(t)")

        # Tab 3: Análisis adicional
        tab3 = self.create_analysis_tab()
        tabs.addTab(tab3, "Análisis Adicional")

        # Tab 4: Métricas
        tab4 = self.create_metrics_tab()
        tabs.addTab(tab4, "Métricas")

        # Tab 5: Resultados
        tab5 = self.create_console_tab()
        tabs.addTab(tab5, "Resultados")

    def create_params_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Grupo de parámetros
        params_group = QGroupBox("Parámetros del Sistema")
        params_layout = QFormLayout()

        self.lambda_input = QLineEdit(str(self.lambda_rate))
        self.mu_p_input = QLineEdit(str(self.mu_p))
        self.mu_i_input = QLineEdit(str(self.mu_i))
        self.periods_input = QLineEdit(str(self.n_periods))

        params_layout.addRow("λ (llegadas/min):", self.lambda_input)
        params_layout.addRow("μ Paciente (servicio/min):", self.mu_p_input)
        params_layout.addRow("μ Impaciente (servicio/min):", self.mu_i_input)
        params_layout.addRow("Número de turistas:", self.periods_input)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Matriz de Markov
        markov_group = QGroupBox("Matriz de Transición")
        markov_layout = QVBoxLayout()
        markov_info = QLabel(
            "P = [[0.7, 0.3], [0.5, 0.5]]\nEstados: [Paciente, Impaciente]"
        )
        markov_layout.addWidget(markov_info)
        markov_group.setLayout(markov_layout)
        layout.addWidget(markov_group)

        # Botón
        self.sim_button = QPushButton("Ejecutar Simulación")
        self.sim_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.sim_button)

        # Status
        self.status_label = QLabel("Listo")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addStretch()
        return widget

    def create_graph_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas)

        return widget

    def create_analysis_tab(self):
        """Crea tab con gráficos adicionales"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Canvas con 4 subplots (2x2)
        self.analysis_canvas = MultiplotCanvas(
            self, nrows=2, ncols=2, width=12, height=10, dpi=90
        )

        # Usar scroll area para gráficos grandes
        scroll = QScrollArea()
        scroll.setWidget(self.analysis_canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        return widget

    def create_metrics_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.metrics_table = QTableWidget()
        layout.addWidget(self.metrics_table)

        return widget

    def create_console_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        return widget

    def run_simulation(self):
        try:
            self.lambda_rate = float(self.lambda_input.text())
            self.mu_p = float(self.mu_p_input.text())
            self.mu_i = float(self.mu_i_input.text())
            self.n_periods = int(self.periods_input.text())

            if (
                self.lambda_rate <= 0
                or self.mu_p <= 0
                or self.mu_i <= 0
                or self.n_periods <= 0
            ):
                raise ValueError("Los valores deben ser positivos")

        except ValueError:
            self.status_label.setText("Error: parámetros inválidos")
            return

        self.sim_button.setEnabled(False)
        self.status_label.setText("Simulando...")

        self.thread = SimulationThread(
            self.lambda_rate, self.mu_p, self.mu_i, self.P, self.n_periods
        )
        self.thread.finished.connect(self.on_simulation_finished)
        self.thread.start()

    def on_simulation_finished(self, results, pi_dist, metrics):
        # Guardar resultados
        self.last_results = results
        self.last_pi_dist = pi_dist
        self.last_metrics = metrics

        # Gráfico principal N(t)
        self.canvas.axes.clear()
        self.canvas.axes.plot(
            results["turista"], results["n_sistema"], "k-", linewidth=1
        )
        self.canvas.axes.set_xlabel("Turista")
        self.canvas.axes.set_ylabel("N(t)")
        self.canvas.axes.set_title("Número de turistas en el sistema")
        self.canvas.axes.grid(True, alpha=0.3)
        self.canvas.draw()

        # Gráficos de análisis adicional
        self.update_analysis_plots(results, pi_dist, metrics)

        # Actualizar tabla
        self.update_metrics_table(pi_dist, metrics, results)

        # Obtener métricas simuladas
        L_sim = results.attrs.get("L_simulated", 0)
        W_sim = results.attrs.get("W_simulated", 0)
        Lq_sim = results.attrs.get("Lq_simulated", 0)
        Wq_sim = results.attrs.get("Wq_simulated", 0)

        # Calcular errores
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

        # Resultados
        output = f"""
Distribución estacionaria:
  π Paciente:    {pi_dist["Paciente"]:.4f}
  π Impaciente:  {pi_dist["Impaciente"]:.4f}

Métricas teóricas:
  ρ:   {metrics["rho"]:.4f}
  L:   {metrics["L"]:.2f}
  W:   {metrics["W"]:.2f} min
  Lq:  {metrics["Lq"]:.2f}
  Wq:  {metrics["Wq"]:.2f} min

Métricas simuladas:
  L:   {L_sim:.2f}  (error: {error_L:.1f}%)
  W:   {W_sim:.2f} min  (error: {error_W:.1f}%)
  Lq:  {Lq_sim:.2f}  (error: {error_Lq:.1f}%)
  Wq:  {Wq_sim:.2f} min  (error: {error_Wq:.1f}%)

Estadísticas:
  N(t) promedio:         {results["n_sistema"].mean():.2f}
  N(t) estado estacionario: {results.attrs.get("N_steady_state", 0):.2f}
  N(t) máximo:           {results["n_sistema"].max()}

Distribución de tipos:
{results["tipo"].value_counts(normalize=True).to_string()}
        """
        self.results_text.setText(output)

        self.sim_button.setEnabled(True)
        self.status_label.setText("Completado")

    def update_analysis_plots(self, results, pi_dist, metrics):
        """Actualiza los 4 gráficos de análisis adicional"""

        # Limpiar todos los ejes
        for ax in self.analysis_canvas.axes.flat:
            ax.clear()

        # Subplot 1: Histograma de tiempos de espera
        ax1 = self.analysis_canvas.axes[0, 0]
        wait_times = results["tiempo_espera"]
        ax1.hist(wait_times, bins=30, color="gray", edgecolor="black", alpha=0.7)
        ax1.set_xlabel("Tiempo de espera (min)")
        ax1.set_ylabel("Frecuencia")
        ax1.set_title("Distribución de Tiempos de Espera")
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Evolución de tipos de turistas
        ax2 = self.analysis_canvas.axes[0, 1]
        # Crear ventanas deslizantes para suavizar
        window_size = max(50, len(results) // 20)
        paciente_rolling = (
            (results["tipo"] == "Paciente").rolling(window=window_size).mean()
        )
        ax2.plot(results["turista"], paciente_rolling, "k-", linewidth=1)
        ax2.axhline(
            y=pi_dist["Paciente"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label="π Paciente",
        )
        ax2.set_xlabel("Turista")
        ax2.set_ylabel("Proporción de Pacientes")
        ax2.set_title(f"Evolución de Tipos (ventana={window_size})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Subplot 3: Comparación de tiempos de servicio por tipo
        ax3 = self.analysis_canvas.axes[1, 0]
        paciente_times = results[results["tipo"] == "Paciente"]["tiempo_servicio"]
        impaciente_times = results[results["tipo"] == "Impaciente"]["tiempo_servicio"]

        bp = ax3.boxplot(
            [paciente_times, impaciente_times],
            labels=["Paciente", "Impaciente"],
            patch_artist=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("lightgray")
        ax3.set_ylabel("Tiempo de servicio (min)")
        ax3.set_title("Tiempos de Servicio por Tipo")
        ax3.grid(True, alpha=0.3, axis="y")

        # Subplot 4: N(t) en estado estacionario (últimos 20%)
        ax4 = self.analysis_canvas.axes[1, 1]
        steady_start = int(0.8 * len(results))
        steady_results = results[steady_start:]
        ax4.plot(
            steady_results["turista"], steady_results["n_sistema"], "k-", linewidth=1
        )
        mean_steady = steady_results["n_sistema"].mean()
        ax4.axhline(
            y=mean_steady,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Promedio={mean_steady:.2f}",
        )
        ax4.set_xlabel("Turista")
        ax4.set_ylabel("N(t)")
        ax4.set_title("N(t) en Estado Estacionario (últimos 20%)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Ajustar espaciado
        self.analysis_canvas.fig.tight_layout()
        self.analysis_canvas.draw()

    def update_metrics_table(self, pi_dist, metrics, results):
        L_sim = results.attrs.get("L_simulated", 0)
        W_sim = results.attrs.get("W_simulated", 0)
        Lq_sim = results.attrs.get("Lq_simulated", 0)
        Wq_sim = results.attrs.get("Wq_simulated", 0)

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

        data = [
            ("Distribución estacionaria", "", "", ""),
            ("π Paciente", f"{pi_dist['Paciente']:.4f}", "", ""),
            ("π Impaciente", f"{pi_dist['Impaciente']:.4f}", "", ""),
            ("", "", "", ""),
            ("Utilización", "", "", ""),
            ("ρ", f"{metrics['rho']:.4f}", "", ""),
            ("", "", "", ""),
            ("Clientes en sistema (L)", "", "", ""),
            ("Teórico", f"{metrics['L']:.2f}", "", "clientes"),
            ("Simulado", f"{L_sim:.2f}", f"{error_L:.1f}%", "clientes"),
            ("", "", "", ""),
            ("Tiempo en sistema (W)", "", "", ""),
            ("Teórico", f"{metrics['W']:.2f}", "", "minutos"),
            ("Simulado", f"{W_sim:.2f}", f"{error_W:.1f}%", "minutos"),
            ("", "", "", ""),
            ("Clientes en cola (Lq)", "", "", ""),
            ("Teórico", f"{metrics['Lq']:.2f}", "", "clientes"),
            ("Simulado", f"{Lq_sim:.2f}", f"{error_Lq:.1f}%", "clientes"),
            ("", "", "", ""),
            ("Tiempo en cola (Wq)", "", "", ""),
            ("Teórico", f"{metrics['Wq']:.2f}", "", "minutos"),
            ("Simulado", f"{Wq_sim:.2f}", f"{error_Wq:.1f}%", "minutos"),
        ]

        self.metrics_table.setRowCount(len(data))
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Métrica", "Valor", "Error", "Unidad"]
        )

        for i, row in enumerate(data):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.metrics_table.setItem(i, j, item)

        # Ajustar columnas
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
