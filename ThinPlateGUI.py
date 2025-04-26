import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QComboBox,
    QStatusBar, QSizePolicy, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import io

# Import analysis logic from the refactored ThinPlate.py
try:
    import ThinPlate as tp 
except ImportError:
    print("Error: ThinPlate.py not found or contains errors.")
    sys.exit(1)

# Define findMax locally within the GUI script
def findMax(array, plate, n_nodesX, n_nodesY):
    """Finds the maximum absolute value and its location in the array."""
    maxVal = np.amax(np.abs(array))
    maxVal_ind = np.unravel_index(np.abs(array).argmax(), array.shape)
    # Calculate coordinates based on indices and plate dimensions
    x_coord = maxVal_ind[1] * plate.a / (n_nodesX - 1) if n_nodesX > 1 else 0
    y_coord = maxVal_ind[0] * plate.b / (n_nodesY - 1) if n_nodesY > 1 else 0
    return {"val": maxVal, "x": x_coord, "y": y_coord}

class MplCanvas(FigureCanvas):
    """Base Matplotlib canvas widget."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)

class MplCanvas2D(MplCanvas):
    """Matplotlib canvas widget for 2D plots."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)
        self.axes = self.fig.add_subplot(111)

class MplCanvasMulti(MplCanvas):
    """Matplotlib canvas widget for multiple subplots."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)

class MplCanvas3D(MplCanvas):
    """Matplotlib canvas widget for 3D plots."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')

class ConvergenceCanvas(FigureCanvas):
    """Matplotlib canvas widget specifically for convergence plots."""
    def __init__(self, parent=None, width=10, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = self.fig.add_subplot(121) # m-convergence
        self.ax2 = self.fig.add_subplot(122) # k-convergence
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)

class AnalysisThread(QThread):
    """Worker thread for running the analysis."""
    analysis_complete = pyqtSignal(object, object, object) 
    error_occurred = pyqtSignal(str)
    convergence_warning = pyqtSignal(str)

    def __init__(self, plate_params, analysis_params, node_params):
        super().__init__()
        self.plate_params = plate_params
        self.analysis_params = analysis_params
        self.node_params = node_params

    def run(self):
        """Execute the analysis in the background and capture warnings."""
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            plate = tp.Plate(**self.plate_params)
            
            n_nodesX = self.node_params['n_nodesX']
            n_nodesY = self.node_params['n_nodesY']

            results, m_hist, k_hist = tp.Analyze(plate, n_nodesX, n_nodesY, **self.analysis_params)
            
            sys.stdout = old_stdout

            output_str = redirected_output.getvalue()
            warnings = [line for line in output_str.splitlines() if "Warning:" in line]
            if warnings:
                self.convergence_warning.emit("\n".join(warnings))

            self.analysis_complete.emit(results, m_hist, k_hist)
        except Exception as e:
            sys.stdout = old_stdout
            self.error_occurred.emit(f"Analysis Error: {e}\nCaptured Output:\n{redirected_output.getvalue()}")
        finally:
            if sys.stdout == redirected_output:
                sys.stdout = old_stdout

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thin Plate Solver GUI")
        self.setGeometry(100, 100, 1200, 750)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.results_data = None
        self.current_plate = None 
        self.current_n_nodesX = 0
        self.current_n_nodesY = 0
        self.results_colorbar = None

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.input_tab = QWidget()
        self.tabs.addTab(self.input_tab, "Input Parameters")
        self.input_layout = QGridLayout(self.input_tab)

        self.plate_canvas = MplCanvasMulti(self, width=6, height=5, dpi=100)
        self.input_layout.addWidget(self.plate_canvas, 0, 1, 5, 1)

        self.inputs = {}
        input_groups_data = {
            "Geometry": {
                "a": ("Plate Length (x, m):", "1.0", "Length of the plate along the x-axis."),
                "b": ("Plate Width (y, m):", "1.0", "Width of the plate along the y-axis."),
                "t0": ("Thickness @ center (m):", "0.05", "Plate thickness at y = b/2."),
                "e": ("Thickness Variation (e):", "0.2", "Parameter controlling linear thickness variation (0 <= e < 1).\nt(y) = t0 * (1 + e * (2*y/b - 1))"),
            },
            "Material": {
                "E": ("Elastic Modulus (Pa):", "200e9", "Young's Modulus of the plate material."),
                "nu": ("Poisson's Ratio:", "0.3", "Poisson's ratio of the plate material (unitless)."),
            },
            "Loading": {
                "Q": ("Load Intensity (Pa):", "-10e3", "Uniformly distributed transverse load intensity (N/m^2).\nNegative value indicates downward load."),
            },
            "Analysis Control": {
                "N_max": ("Max Fourier Terms (m):", "100", "Maximum number of terms in the Fourier series expansion."),
                "k_max": ("Max Perturbation Terms (k):", "10", "Maximum order for the perturbation series."),
                "m_tol": ("Fourier Tolerance:", "1e-6", "Relative tolerance for Fourier series convergence."),
                "k_tol": ("Perturbation Tolerance:", "1e-6", "Relative tolerance for perturbation series convergence."),
            },
            "Discretization": {
                "node_density": ("Node Density (/m):", "50", "Number of nodes per meter used for calculations and plotting.\nMinimum 21 nodes enforced in each direction.")
            }
        }

        group_row = 0
        live_update_inputs = ["a", "b", "e", "t0"]
        
        for group_title, params in input_groups_data.items():
            group_box = QGroupBox(group_title)
            group_layout = QGridLayout(group_box)
            param_row = 0
            for name, (label_text, default_val, tooltip_text) in params.items():
                label = QLabel(label_text)
                label.setToolTip(tooltip_text)
                line_edit = QLineEdit(default_val)
                line_edit.setToolTip(tooltip_text)
                
                group_layout.addWidget(label, param_row, 0)
                group_layout.addWidget(line_edit, param_row, 1)
                
                self.inputs[name] = line_edit
                if name in live_update_inputs:
                    line_edit.textChanged.connect(self._update_live_plate_plot)
                param_row += 1
            
            self.input_layout.addWidget(group_box, group_row, 0)
            group_row += 1

        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.input_layout.addWidget(self.run_button, group_row, 0)

        self.input_layout.setColumnStretch(0, 1)
        self.input_layout.setColumnStretch(1, 2)

        self._update_live_plate_plot()

        self.results_tab = QWidget()
        self.tabs.addTab(self.results_tab, "Results (2D)")
        self.results_layout = QVBoxLayout(self.results_tab)
        
        self.results_combo = QComboBox()
        self.results_combo.addItems(tp.titles)
        self.results_combo.currentIndexChanged.connect(self.update_results_plot)
        self.results_combo.currentIndexChanged.connect(self.update_3d_plot)
        
        self.max_value_label = QLabel("Max Value: N/A")
        self.max_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        results_top_layout = QHBoxLayout()
        results_top_layout.addWidget(QLabel("Select Result:"))
        results_top_layout.addWidget(self.results_combo)
        results_top_layout.addWidget(self.max_value_label)
        self.results_layout.addLayout(results_top_layout)

        self.results_canvas = MplCanvas2D(self, width=6, height=5, dpi=100)
        self.results_layout.addWidget(self.results_canvas)

        self.plot_3d_tab = QWidget()
        self.tabs.addTab(self.plot_3d_tab, "Results (3D)")
        self.plot_3d_layout = QVBoxLayout(self.plot_3d_tab)
        self.plot_3d_canvas = MplCanvas3D(self, width=7, height=6, dpi=100)
        self.plot_3d_layout.addWidget(self.plot_3d_canvas)
        self.plot_3d_canvas.axes.text(0.5, 0.5, 0.5, "Run analysis to view 3D plot", 
                                      ha='center', va='center', transform=self.plot_3d_canvas.axes.transAxes)

        self.convergence_tab = QWidget()
        self.tabs.addTab(self.convergence_tab, "Convergence")
        self.convergence_layout = QVBoxLayout(self.convergence_tab)
        self.convergence_canvas = ConvergenceCanvas(self, width=10, height=5, dpi=100)
        self.convergence_layout.addWidget(self.convergence_canvas)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.analysis_thread = None

    def get_input_params(self):
        """Reads and validates input parameters."""
        params = {}
        plate_keys = ["a", "b", "e", "t0", "E", "nu", "Q"]
        analysis_keys = ["N_max", "k_max", "m_tol", "k_tol"]
        node_keys = ["node_density"]
        
        try:
            for name, widget in self.inputs.items():
                params[name] = float(widget.text())
            
            params["N_max"] = int(params["N_max"])
            params["k_max"] = int(params["k_max"])
            params["node_density"] = int(params["node_density"])

            plate_params = {k: params[k] for k in plate_keys}
            analysis_params = {k: params[k] for k in analysis_keys}
            node_params_in = {k: params[k] for k in node_keys}

            n_nodesX = int(plate_params['a'] * node_params_in['node_density']) + 1
            n_nodesY = int(plate_params['b'] * node_params_in['node_density']) + 1
            n_nodesX = max(n_nodesX, 21)
            n_nodesY = max(n_nodesY, 21)
            node_params_out = {'n_nodesX': n_nodesX, 'n_nodesY': n_nodesY}

            if not (0 <= plate_params['e'] < 1):
                raise ValueError("Thickness variation 'e' must be in the range [0, 1).")

            return plate_params, analysis_params, node_params_out
        except ValueError as e:
            self.statusBar.showMessage(f"Input Error: Invalid number format or value - {e}")
            return None, None, None
        except KeyError as e:
            self.statusBar.showMessage(f"Input Error: Missing input field - {e}")
            return None, None, None

    def run_analysis(self):
        """Starts the analysis in a background thread."""
        plate_params, analysis_params, node_params = self.get_input_params()
        if plate_params is None:
            return

        self.statusBar.showMessage("Running analysis...")
        self.run_button.setEnabled(False)

        self.analysis_thread = AnalysisThread(plate_params, analysis_params, node_params)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.error_occurred.connect(self.on_analysis_error)
        self.analysis_thread.convergence_warning.connect(self.on_convergence_warning)
        self.analysis_thread.start()

    def on_analysis_error(self, error_message):
        """Handles errors reported by the analysis thread."""
        self.statusBar.showMessage("Analysis failed.")
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.run_button.setEnabled(True)

    def on_convergence_warning(self, warning_message):
        """Shows a warning message if convergence was not met."""
        self.statusBar.showMessage("Analysis complete with convergence warnings.")
        QMessageBox.warning(self, "Convergence Warning", warning_message)

    def on_analysis_complete(self, results, m_history, k_history):
        """Handles the results when analysis finishes successfully."""
        if "warnings" not in self.statusBar.currentMessage():
            self.statusBar.showMessage("Analysis complete. Displaying results.")
        self.run_button.setEnabled(True)

        self.results_data = results
        plate_params, analysis_params, node_params = self.get_input_params()
        if plate_params:
            self.current_plate = tp.Plate(**plate_params)
            self.current_n_nodesX = node_params['n_nodesX']
            self.current_n_nodesY = node_params['n_nodesY']
            self._update_live_plate_plot()
        else:
            self.current_plate = None
            self.current_n_nodesX = 0
            self.current_n_nodesY = 0

        self.update_results_plot()
        self.update_3d_plot()
        self.update_convergence_plot(m_history, k_history, analysis_params['m_tol'], analysis_params['k_tol'], plate_params['e'])
        
        self.tabs.setCurrentWidget(self.results_tab)

    def _update_live_plate_plot(self):
        """Updates the plate geometry plot live based on input fields."""
        try:
            a = float(self.inputs['a'].text())
            b = float(self.inputs['b'].text())
            e = float(self.inputs['e'].text())
            t0 = float(self.inputs['t0'].text())

            if a <= 0 or b <= 0 or t0 <= 0:
                raise ValueError("Dimensions (a, b, t0) must be positive.")
        except (ValueError, KeyError) as err:
            fig = self.plate_canvas.fig
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Invalid Input:\n{err}', 
                    ha='center', va='center', color='red', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.plate_canvas.draw()
            return

        fig = self.plate_canvas.fig
        fig.clear()

        ax_top = fig.add_subplot(1, 2, 1) # Left plot for Top View
        ax_side = fig.add_subplot(1, 2, 2) # Right plot for Side View

        ax_top.plot([0, a, a, 0, 0], [0, 0, b, b, 0], 'b-') 
        ax_top.set_xlabel("x (m)")
        ax_top.set_ylabel("y (m)")
        ax_top.set_title(f"Top View (a={a:.2f}, b={b:.2f})")
        ax_top.set_xlim(-0.1*a, 1.1*a)
        ax_top.set_ylim(-0.1*b, 1.1*b)
        ax_top.grid(True)

        y_plot = np.linspace(0, b, 100)
        try:
            thickness_y = t0 * (1 + e * (2 * y_plot / b - 1)) if b > 0 else np.full_like(y_plot, t0)
        except ZeroDivisionError:
            thickness_y = np.full_like(y_plot, t0)

        ax_side.plot(y_plot, thickness_y, 'r-') 
        ax_side.fill_between(y_plot, 0, thickness_y, color='red', alpha=0.3) 
        
        ax_side.set_xlabel("y (m)")
        ax_side.set_ylabel("Thickness t(y) (m)")
        ax_side.set_title(f"Side View - Thickness Profile (t0={t0:.3f}, e={e:.2f})")
        ax_side.grid(True)
        ax_side.set_xlim(ax_top.get_ylim()) 
        min_t = t0 * (1 - abs(e)) 
        max_t = t0 * (1 + abs(e))
        padding_t = (max_t - min_t) * 0.1 + 1e-6 
        ax_side.set_ylim(bottom=max(0, min_t - padding_t), top=max_t + padding_t) 

        try:
            fig.tight_layout(w_pad=3.0) 
        except ValueError:
            pass
        self.plate_canvas.draw()

    def update_results_plot(self):
        """Updates the results plot based on dropdown selection."""
        ax = self.results_canvas.axes
        fig = self.results_canvas.fig

        if self.results_colorbar:
            try:
                self.results_colorbar.remove()
            except Exception as e:
                print(f"Ignoring error removing old colorbar: {e}")
            self.results_colorbar = None

        ax.clear()

        if self.results_data is None or self.current_plate is None:
            ax.set_title("No Results Available")
            self.results_canvas.draw()
            self.max_value_label.setText("Max Value: N/A")
            return

        selected_title = self.results_combo.currentText()
        result_index = tp.title_map.get(selected_title)

        if result_index is None:
            ax.set_title("Invalid Result Selection")
            self.results_canvas.draw()
            return

        data_to_plot = self.results_data[result_index]
        
        im = ax.imshow(data_to_plot, cmap='Spectral_r', origin='lower', 
                       extent=[0, self.current_plate.a, 0, self.current_plate.b],
                       aspect='auto') 
        
        self.results_colorbar = fig.colorbar(im, ax=ax, label=selected_title) 

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Result: {selected_title}")
        try:
            fig.tight_layout()
        except ValueError:
            pass
        self.results_canvas.draw()

        try:
            max_info = findMax(data_to_plot, self.current_plate, self.current_n_nodesX, self.current_n_nodesY)
            self.max_value_label.setText(f"Max Abs: {max_info['val']:.4g} @ (x={max_info['x']:.3f}, y={max_info['y']:.3f})")
        except Exception as e:
            self.max_value_label.setText("Max Value: Error")
            print(f"Error finding max value: {e}")

    def update_3d_plot(self):
        """Updates the 3D results plot based on dropdown selection."""
        ax = self.plot_3d_canvas.axes
        fig = self.plot_3d_canvas.fig
        ax.clear()

        if self.results_data is None or self.current_plate is None or self.current_n_nodesX < 2 or self.current_n_nodesY < 2:
            ax.text(0.5, 0.5, 0.5, "Run analysis to view 3D plot", 
                    ha='center', va='center', transform=ax.transAxes)
            self.plot_3d_canvas.draw()
            return

        selected_title = self.results_combo.currentText()
        result_index = tp.title_map.get(selected_title)

        if result_index is None:
            ax.text(0.5, 0.5, 0.5, "Invalid Result Selection", 
                    ha='center', va='center', transform=ax.transAxes)
            self.plot_3d_canvas.draw()
            return

        data_to_plot = self.results_data[result_index]

        x = np.linspace(0, self.current_plate.a, self.current_n_nodesX)
        y = np.linspace(0, self.current_plate.b, self.current_n_nodesY)
        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(X, Y, data_to_plot, cmap='Spectral_r', 
                               linewidth=0, antialiased=False)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel(selected_title)
        ax.set_title(f"3D View: {selected_title}")

        try:
            fig.tight_layout()
        except ValueError:
            pass
        self.plot_3d_canvas.draw()

    def update_convergence_plot(self, m_history, k_history, m_tol, k_tol, e_val):
        """Updates the convergence plot."""
        canvas = self.convergence_canvas
        ax1 = canvas.ax1
        ax2 = canvas.ax2
        ax1.clear()
        ax2.clear()
        
        canvas.fig.suptitle(f'Convergence History for e = {e_val:.2f}')

        # --- Plot m-convergence ---
        min_m_err = 1.0 
        max_m_err = 1e-16 
        max_m_val = 0 # Track the maximum m value plotted
        for k_idx, k_m_history in enumerate(m_history):
            if k_m_history: 
                m_vals, rel_errs_m = zip(*k_m_history)
                if m_vals: # Check if there are m values for this k
                    max_m_val = max(max_m_val, max(m_vals)) # Update overall max m
                
                rel_errs_m_plot = []
                for err in rel_errs_m:
                    plot_err = max(err, 1e-16) 
                    rel_errs_m_plot.append(plot_err)
                    if plot_err > 1e-16: 
                         min_m_err = min(min_m_err, plot_err)
                         max_m_err = max(max_m_err, plot_err) 

                ax1.plot(m_vals, rel_errs_m_plot, marker='o', markersize=4, linestyle='-', label=f'k={k_idx}')
        
        ax1.axhline(m_tol, color='r', linestyle='--', label=f'm_tol = {m_tol:.1e}')
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=min(m_tol / 100, min_m_err / 10, 1e-9)) 
        ax1.set_xlabel('Fourier Terms (m)')
        ax1.set_ylabel('Relative Error')
        ax1.set_title('Fourier Series (m) Convergence')
        # Set integer ticks for m-axis
        if max_m_val > 0:
            nbins_m = max(int(max_m_val / 2) + 1, 5) if max_m_val < 10 else 'auto' 
            ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=nbins_m))
            ax1.set_xlim(left=0, right=max_m_val + 1) 

        ax1.legend()
        ax1.grid(True, which="both", ls="--")

        # --- Plot k-convergence ---
        min_k_err = 1.0 
        max_k_err = 1e-16 
        if k_history:
            k_vals, rel_errs_k = zip(*k_history)
            k_vals_plot = []
            rel_errs_k_plot = []
            for k, err in k_history:
                 if err > 0: 
                     plot_err = max(err, 1e-16) 
                     k_vals_plot.append(k)
                     rel_errs_k_plot.append(plot_err)
                     if plot_err > 1e-16: 
                         min_k_err = min(min_k_err, plot_err)
                         max_k_err = max(max_k_err, plot_err) 
            
            if k_vals_plot:
                ax2.plot(k_vals_plot, rel_errs_k_plot, marker='s', markersize=4, linestyle='-')

        ax2.axhline(k_tol, color='r', linestyle='--', label=f'k_tol = {k_tol:.1e}')
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=min(k_tol / 100, min_k_err / 10, 1e-9)) 
        ax2.set_xlabel('Perturbation Order (k)')
        ax2.set_ylabel('Relative Error')
        ax2.set_title('Perturbation Series (k) Convergence')
        ax2.legend()
        ax2.grid(True, which="both", ls="--")
        
        if k_history:
            plotted_k_vals = [k for k, err in k_history if err > 0]
            if plotted_k_vals:
                max_k_val = max(plotted_k_vals)
                ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=max(max_k_val + 1, 5)))

        canvas.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
