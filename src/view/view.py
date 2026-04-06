"""Vista del programa — interfaz rediseñada con sidebar azul oscuro.
- Autor: Alejandra Michelle Mateo Garcia & Leyva Triana Isis Valeria
- Fecha: 20 de febrero del 2026
- Versión: 3.0
- Cambios:
    ✓ Sidebar azul oscuro tipo VS Code con secciones colapsables
    ✓ Panel de comparación Original vs Resultado lado a lado
    ✓ Histograma en vivo embebido (matplotlib en canvas Qt)
    ✓ Barra de estado con info de imagen (nombre, dimensiones, modo)
    ✓ Arquitectura extensible: agregar secciones con _make_section()
- Escuela: ESCOM-IPN
- Materia: Análisis de Imágenes
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QScrollArea, QSplitter,
    QSizePolicy, QFrame, QToolButton, QApplication,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ──────────────────────────────────────────────────────────────────────────────
# Paleta de colores centralizada — cambia aquí para recolorear toda la app
# ──────────────────────────────────────────────────────────────────────────────
SIDEBAR_BG      = "#1e2b3c"   # azul oscuro principal
SIDEBAR_HOVER   = "#263548"   # hover sobre ítem
SIDEBAR_ACTIVE  = "#2d4a6b"   # ítem activo
SIDEBAR_BORDER  = "#2a3f57"   # separadores internos
SIDEBAR_TEXT    = "#a8b8cc"   # texto normal
SIDEBAR_TEXT_HI = "#e8f0f8"   # texto activo / destacado
SIDEBAR_LABEL   = "#5a7a99"   # etiquetas de sección
ACCENT          = "#4a9eff"   # azul acento (botones primarios)
ACCENT_HOVER    = "#3a8eef"

CANVAS_BG       = "#1a1a2e"   # fondo del área de imagen
TOOLBAR_BG      = "#f5f6f8"
STATUS_BG       = "#f0f2f5"

# ──────────────────────────────────────────────────────────────────────────────
# ImageViewer
# ──────────────────────────────────────────────────────────────────────────────
class ImageViewer(QLabel):
    """Visor de imagen con soporte para escalado y texto placeholder."""

    def __init__(self, placeholder: str = "Sin imagen"):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {CANVAS_BG}; border: none;")
        self._placeholder = placeholder
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText(f'<span style="color:#3a4a6b; font-size:13px;">{self._placeholder}</span>')

    def set_image(self, cv_img):
        if cv_img is None:
            self._show_placeholder()
            return
        h, w = cv_img.shape[:2]
        if cv_img.ndim == 2:
            q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            q_img = QImage(cv_img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-escalar al cambiar tamaño si hay una imagen
        if self.pixmap() and not self.pixmap().isNull():
            scaled = self.pixmap().scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)


# ──────────────────────────────────────────────────────────────────────────────
# LiveHistogramCanvas — histograma embebido con matplotlib
# ──────────────────────────────────────────────────────────────────────────────
class LiveHistogramCanvas(FigureCanvas):
    """Canvas de matplotlib integrado en Qt para histograma en vivo."""

    def __init__(self):
        self.fig = Figure(figsize=(3, 1.6), dpi=90, facecolor=SIDEBAR_BG)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(180)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.10)
        self._style_axes()

    def _style_axes(self):
        self.ax.set_facecolor(SIDEBAR_BG)
        self.ax.tick_params(colors=SIDEBAR_LABEL, labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color(SIDEBAR_BORDER)
        self.ax.set_xlabel("", fontsize=7)

    def update_histogram(self, hists: dict | None, mode: str = "rgb"):
        """Recibe dict con claves 'r','g','b' o array de grises."""
        self.ax.clear()
        self._style_axes()

        if hists is None:
            self.ax.text(
                0.5, 0.5, "carga una imagen",
                transform=self.ax.transAxes,
                color=SIDEBAR_LABEL, fontsize=8, ha="center", va="center",
            )
        elif mode == "gray" and isinstance(hists, np.ndarray):
            self.ax.fill_between(range(256), hists, color="#a8b8cc", alpha=0.6)
            self.ax.plot(hists, color="#e8f0f8", linewidth=0.8)
        else:
            for canal, color in [("r", "#ff6b6b"), ("g", "#6bffa0"), ("b", "#6bb5ff")]:
                if canal in hists:
                    self.ax.plot(hists[canal], color=color, linewidth=0.9, alpha=0.85)

        self.ax.set_xlim(0, 255)
        self.ax.set_yticks([])
        self.draw()


# ──────────────────────────────────────────────────────────────────────────────
# SidebarSection — sección colapsable del sidebar
# ──────────────────────────────────────────────────────────────────────────────
class SidebarSection(QWidget):
    """Sección colapsable con título y contenido interno."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._collapsed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Cabecera clicable
        self.header = QToolButton()
        self.header.setText(f"  {title}")
        self.header.setCheckable(False)
        self.header.setArrowType(Qt.DownArrow)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                color: {SIDEBAR_LABEL};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                border: none;
                padding: 8px 14px 4px;
                text-align: left;
            }}
            QToolButton:hover {{ color: {SIDEBAR_TEXT}; }}
        """)
        self.header.clicked.connect(self._toggle)
        layout.addWidget(self.header)

        # Contenedor de contenido
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 4)
        self.body_layout.setSpacing(0)
        layout.addWidget(self.body)

        # Separador inferior
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {SIDEBAR_BORDER}; max-height: 1px;")
        layout.addWidget(sep)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self.body.setVisible(not self._collapsed)
        self.header.setArrowType(
            Qt.RightArrow if self._collapsed else Qt.DownArrow
        )

    def add_widget(self, widget: QWidget):
        self.body_layout.addWidget(widget)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de widgets estilizados
# ──────────────────────────────────────────────────────────────────────────────
def _sidebar_button(text: str, accent: bool = False) -> QPushButton:
    btn = QPushButton(text)
    if accent:
        style = f"""
            QPushButton {{
                background: {ACCENT};
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 6px 12px;
                font-size: 12px;
                margin: 2px 12px;
            }}
            QPushButton:hover {{ background: {ACCENT_HOVER}; }}
            QPushButton:pressed {{ background: #2a7edf; }}
            QPushButton:disabled {{ background: {SIDEBAR_BORDER}; color: {SIDEBAR_LABEL}; }}
        """
    else:
        style = f"""
            QPushButton {{
                background: transparent;
                color: {SIDEBAR_TEXT};
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 12px;
                text-align: left;
            }}
            QPushButton:hover {{
                background: {SIDEBAR_HOVER};
                color: {SIDEBAR_TEXT_HI};
            }}
            QPushButton:pressed {{ background: {SIDEBAR_ACTIVE}; }}
            QPushButton:disabled {{ color: {SIDEBAR_LABEL}; }}
        """
    btn.setStyleSheet(style)
    return btn


def _sidebar_combo(items: list[str]) -> QComboBox:
    combo = QComboBox()
    combo.addItems(items)
    combo.setStyleSheet(f"""
        QComboBox {{
            background: {SIDEBAR_ACTIVE};
            color: {SIDEBAR_TEXT_HI};
            border: 1px solid {SIDEBAR_BORDER};
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            margin: 2px 12px;
        }}
        QComboBox::drop-down {{ border: none; width: 20px; }}
        QComboBox QAbstractItemView {{
            background: {SIDEBAR_BG};
            color: {SIDEBAR_TEXT};
            selection-background-color: {SIDEBAR_ACTIVE};
        }}
    """)
    return combo


def _mini_button(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(26)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {SIDEBAR_ACTIVE};
            color: {SIDEBAR_TEXT};
            border: 1px solid {SIDEBAR_BORDER};
            border-radius: 4px;
            font-size: 11px;
            padding: 0 8px;
        }}
        QPushButton:hover {{ background: {SIDEBAR_HOVER}; color: {SIDEBAR_TEXT_HI}; }}
        QPushButton:pressed {{ background: {SIDEBAR_BG}; }}
    """)
    return btn


# ──────────────────────────────────────────────────────────────────────────────
# MainWindow
# ──────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.controller = None
        self._init_ui()

    def set_controller(self, controller):
        self.controller = controller
        self._connect_signals()

    # ── Construcción de UI ───────────────────────────────────────────────────

    def _init_ui(self):
        self.setWindowTitle("Práctica 1 · Análisis de Imagen Digital — ESCOM IPN")
        self.setMinimumSize(1100, 680)
        self.resize(1280, 740)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_main_area(), stretch=1)

    # ── Sidebar ──────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet(f"background: {SIDEBAR_BG};")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo / título
        header = QWidget()
        header.setStyleSheet(f"background: {SIDEBAR_BG}; border-bottom: 1px solid {SIDEBAR_BORDER};")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(14, 12, 14, 10)
        title_lbl = QLabel("Imagen Digital")
        title_lbl.setStyleSheet(f"color: {SIDEBAR_TEXT_HI}; font-size: 14px; font-weight: 600;")
        sub_lbl = QLabel("ESCOM · Análisis de Imágenes")
        sub_lbl.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px;")
        h_layout.addWidget(title_lbl)
        h_layout.addWidget(sub_lbl)
        layout.addWidget(header)

        # Área scrollable de secciones
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {SIDEBAR_BG}; border: none; }}
            QScrollBar:vertical {{
                background: {SIDEBAR_BG}; width: 4px; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {SIDEBAR_BORDER}; border-radius: 2px;
            }}
        """)

        inner = QWidget()
        inner.setStyleSheet(f"background: {SIDEBAR_BG};")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(0)

        # ── Sección: Archivo ──────────────────────────────────────────
        sec_file = SidebarSection("Archivo")
        self.btn_load = _sidebar_button("  Cargar imagen")
        self.btn_save = _sidebar_button("  Guardar resultado")
        self.btn_save.setEnabled(False)
        sec_file.add_widget(self.btn_load)
        sec_file.add_widget(self.btn_save)
        inner_layout.addWidget(sec_file)

        # ── Sección: Canales ──────────────────────────────────────────
        sec_channels = SidebarSection("Canales")
        self.btn_rgb  = _sidebar_button("  Componentes RGB")
        self.btn_gray = _sidebar_button("  Escala de grises")
        sec_channels.add_widget(self.btn_rgb)
        sec_channels.add_widget(self.btn_gray)
        inner_layout.addWidget(sec_channels)

        # ── Sección: Mapa de color ────────────────────────────────────
        sec_map = SidebarSection("Mapa de color")
        self.map_combo = _sidebar_combo([
            "TWILIGHT", "TURBO", "VIRDIS", "PINK",
            "INFERNO", "WINTER", "HSV", "PARULA",
            "PERSONALIZADO (Morado-Fucsia)",
        ])
        self.btn_apply_map = _sidebar_button("  Aplicar mapa", accent=True)
        sec_map.add_widget(self.map_combo)
        sec_map.add_widget(self.btn_apply_map)
        inner_layout.addWidget(sec_map)

        # ── Sección: Binarización ─────────────────────────────────────
        sec_bin = SidebarSection("Binarización")

        slider_row = QWidget()
        slider_row.setStyleSheet("background: transparent;")
        sr_layout = QHBoxLayout(slider_row)
        sr_layout.setContentsMargins(12, 2, 12, 2)
        sr_layout.setSpacing(6)
        lbl_th = QLabel("Umbral")
        lbl_th.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 11px;")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {SIDEBAR_BORDER}; height: 3px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 12px; height: 12px;
                margin: -5px 0; border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
        """)
        self.threshold_label = QLabel("128")
        self.threshold_label.setFixedWidth(28)
        self.threshold_label.setStyleSheet(f"color: {SIDEBAR_TEXT_HI}; font-size: 11px;")
        self.threshold_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sr_layout.addWidget(lbl_th)
        sr_layout.addWidget(self.threshold_slider)
        sr_layout.addWidget(self.threshold_label)
        sec_bin.add_widget(slider_row)

        btn_row = QWidget()
        btn_row.setStyleSheet("background: transparent;")
        br_layout = QHBoxLayout(btn_row)
        br_layout.setContentsMargins(12, 2, 12, 6)
        br_layout.setSpacing(4)
        self.btn_binary_fixed    = _mini_button("Fijo")
        self.btn_binary_otsu     = _mini_button("Otsu")
        self.btn_binary_adaptive = _mini_button("Adaptativo")
        br_layout.addWidget(self.btn_binary_fixed)
        br_layout.addWidget(self.btn_binary_otsu)
        br_layout.addWidget(self.btn_binary_adaptive)
        sec_bin.add_widget(btn_row)
        inner_layout.addWidget(sec_bin)

        # ── Sección: Modelos de color ─────────────────────────────────
        sec_models = SidebarSection("Modelos de color")
        self.models_combo = _sidebar_combo([
            "RGB (Original)",
            "CMYK (Cian, Magenta, Amarillo, Negro)",
            "HSV (Matiz, Saturación, Valor)",
            "HSI (Matiz, Saturación, Intensidad)",
            "YUV (Luminancia, Crominancia)",
            "LAB (CIE L*a*b*)",
            "XYZ (CIE 1931)",
        ])
        self.btn_apply_model = _sidebar_button("  Aplicar modelo", accent=True)
        sec_models.add_widget(self.models_combo)
        sec_models.add_widget(self.btn_apply_model)
        inner_layout.addWidget(sec_models)

        # ── Sección: Práctica 3-a — Ruido ────────────────────────────
        sec_noise = SidebarSection("3-a  Ruido")

        noise_combo_row = QWidget()
        noise_combo_row.setStyleSheet("background: transparent;")
        nc_layout = QHBoxLayout(noise_combo_row)
        nc_layout.setContentsMargins(12, 2, 12, 2)
        nc_layout.setSpacing(6)
        lbl_noise = QLabel("Tipo")
        lbl_noise.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 11px;")
        self.noise_combo = _sidebar_combo(["Sal y pimienta", "Gaussiano"])
        nc_layout.addWidget(lbl_noise)
        nc_layout.addWidget(self.noise_combo)
        sec_noise.add_widget(noise_combo_row)

        noise_slider_row = QWidget()
        noise_slider_row.setStyleSheet("background: transparent;")
        ns_layout = QHBoxLayout(noise_slider_row)
        ns_layout.setContentsMargins(12, 2, 12, 2)
        ns_layout.setSpacing(6)
        lbl_ni = QLabel("Intensidad")
        lbl_ni.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 11px;")
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(1, 100)
        self.noise_slider.setValue(5)
        self.noise_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {SIDEBAR_BORDER}; height: 3px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 12px; height: 12px;
                margin: -5px 0; border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
        """)
        self.noise_label = QLabel("5 %")
        self.noise_label.setFixedWidth(32)
        self.noise_label.setStyleSheet(f"color: {SIDEBAR_TEXT_HI}; font-size: 11px;")
        self.noise_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.noise_slider.valueChanged.connect(
            lambda v: self.noise_label.setText(f"{v} %")
        )
        ns_layout.addWidget(lbl_ni)
        ns_layout.addWidget(self.noise_slider)
        ns_layout.addWidget(self.noise_label)
        sec_noise.add_widget(noise_slider_row)

        noise_btn_row = QWidget()
        noise_btn_row.setStyleSheet("background: transparent;")
        nb_layout = QHBoxLayout(noise_btn_row)
        nb_layout.setContentsMargins(12, 2, 12, 6)
        nb_layout.setSpacing(4)
        self.btn_apply_noise    = _mini_button("Aplicar")
        self.btn_filter_median  = _mini_button("Mediana")
        self.btn_filter_gauss   = _mini_button("Gaussiano")
        nb_layout.addWidget(self.btn_apply_noise)
        nb_layout.addWidget(self.btn_filter_median)
        nb_layout.addWidget(self.btn_filter_gauss)
        sec_noise.add_widget(noise_btn_row)
        inner_layout.addWidget(sec_noise)

        # ── Sección: Práctica 3-b — Operaciones ──────────────────────
        sec_ops = SidebarSection("3-b  Operaciones")

        lbl_arith = QLabel("  Aritmética (escalar)")
        lbl_arith.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px; padding: 4px 0 0;")
        sec_ops.add_widget(lbl_arith)

        scalar_row = QWidget()
        scalar_row.setStyleSheet("background: transparent;")
        sc_layout = QHBoxLayout(scalar_row)
        sc_layout.setContentsMargins(12, 2, 12, 2)
        sc_layout.setSpacing(4)
        lbl_sc = QLabel("Val")
        lbl_sc.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 11px;")
        self.scalar_slider = QSlider(Qt.Horizontal)
        self.scalar_slider.setRange(1, 200)
        self.scalar_slider.setValue(50)
        self.scalar_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {SIDEBAR_BORDER}; height: 3px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 12px; height: 12px;
                margin: -5px 0; border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
        """)
        self.scalar_label = QLabel("50")
        self.scalar_label.setFixedWidth(28)
        self.scalar_label.setStyleSheet(f"color: {SIDEBAR_TEXT_HI}; font-size: 11px;")
        self.scalar_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.scalar_slider.valueChanged.connect(
            lambda v: self.scalar_label.setText(str(v))
        )
        sc_layout.addWidget(lbl_sc)
        sc_layout.addWidget(self.scalar_slider)
        sc_layout.addWidget(self.scalar_label)
        sec_ops.add_widget(scalar_row)

        arith_btn_row = QWidget()
        arith_btn_row.setStyleSheet("background: transparent;")
        ab_layout = QHBoxLayout(arith_btn_row)
        ab_layout.setContentsMargins(12, 2, 12, 4)
        ab_layout.setSpacing(4)
        self.btn_arith_add  = _mini_button("+ Suma")
        self.btn_arith_sub  = _mini_button("− Resta")
        self.btn_arith_mul  = _mini_button("× Multi")
        ab_layout.addWidget(self.btn_arith_add)
        ab_layout.addWidget(self.btn_arith_sub)
        ab_layout.addWidget(self.btn_arith_mul)
        sec_ops.add_widget(arith_btn_row)

        lbl_logic = QLabel("  Lógica (dos imágenes)")
        lbl_logic.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px; padding: 4px 0 0;")
        sec_ops.add_widget(lbl_logic)

        logic_btn_row = QWidget()
        logic_btn_row.setStyleSheet("background: transparent;")
        lb_layout = QHBoxLayout(logic_btn_row)
        lb_layout.setContentsMargins(12, 2, 12, 2)
        lb_layout.setSpacing(4)
        self.btn_logic_and  = _mini_button("AND")
        self.btn_logic_or   = _mini_button("OR")
        self.btn_logic_xor  = _mini_button("XOR")
        self.btn_logic_not  = _mini_button("NOT")
        lb_layout.addWidget(self.btn_logic_and)
        lb_layout.addWidget(self.btn_logic_or)
        lb_layout.addWidget(self.btn_logic_xor)
        lb_layout.addWidget(self.btn_logic_not)
        sec_ops.add_widget(logic_btn_row)

        lbl_rel = QLabel("  Relacional (umbral)")
        lbl_rel.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px; padding: 4px 0 0;")
        sec_ops.add_widget(lbl_rel)

        rel_btn_row = QWidget()
        rel_btn_row.setStyleSheet("background: transparent;")
        rb_layout = QHBoxLayout(rel_btn_row)
        rb_layout.setContentsMargins(12, 2, 12, 6)
        rb_layout.setSpacing(4)
        self.btn_rel_gt  = _mini_button("> Mayor")
        self.btn_rel_lt  = _mini_button("< Menor")
        self.btn_rel_eq  = _mini_button("≈ Igual")
        rb_layout.addWidget(self.btn_rel_gt)
        rb_layout.addWidget(self.btn_rel_lt)
        rb_layout.addWidget(self.btn_rel_eq)
        sec_ops.add_widget(rel_btn_row)
        inner_layout.addWidget(sec_ops)

        # ── Sección: Práctica 3-c — Componentes conexas ───────────────
        sec_cc = SidebarSection("3-c  Conteo de objetos")

        cc_btn_row = QWidget()
        cc_btn_row.setStyleSheet("background: transparent;")
        cc_layout = QHBoxLayout(cc_btn_row)
        cc_layout.setContentsMargins(12, 4, 12, 4)
        cc_layout.setSpacing(4)
        self.btn_cc_v4  = _mini_button("Vecindad 4")
        self.btn_cc_v8  = _mini_button("Vecindad 8")
        self.btn_cc_cmp = _mini_button("Comparar")
        cc_layout.addWidget(self.btn_cc_v4)
        cc_layout.addWidget(self.btn_cc_v8)
        cc_layout.addWidget(self.btn_cc_cmp)
        sec_cc.add_widget(cc_btn_row)
        inner_layout.addWidget(sec_cc)

        # ── Sección: Práctica 4 — Morfología Matemática ──────────────
        sec_morph = SidebarSection("4  Morfología Matemática")

        # EE: forma y tamaño
        ee_row = QWidget()
        ee_row.setStyleSheet("background: transparent;")
        ee_layout = QHBoxLayout(ee_row)
        ee_layout.setContentsMargins(12, 4, 12, 2)
        ee_layout.setSpacing(6)
        lbl_ee = QLabel("EE")
        lbl_ee.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 11px;")
        self.morph_shape_combo = _sidebar_combo(["Rect", "Cruz", "Elipse"])
        self.morph_shape_combo.setFixedWidth(80)
        lbl_sz = QLabel("sz")
        lbl_sz.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 11px;")
        self.morph_size_combo = _sidebar_combo(["3", "5", "7", "9", "11"])
        self.morph_size_combo.setFixedWidth(50)
        ee_layout.addWidget(lbl_ee)
        ee_layout.addWidget(self.morph_shape_combo)
        ee_layout.addWidget(lbl_sz)
        ee_layout.addWidget(self.morph_size_combo)
        ee_layout.addStretch()
        sec_morph.add_widget(ee_row)

        # Modo: Binaria / Grises
        mode_row = QWidget()
        mode_row.setStyleSheet("background: transparent;")
        mr_layout = QHBoxLayout(mode_row)
        mr_layout.setContentsMargins(12, 2, 12, 2)
        mr_layout.setSpacing(4)
        self.btn_morph_bin  = _mini_button("Binaria")
        self.btn_morph_gray = _mini_button("Grises")
        self.btn_morph_all_bin  = _mini_button("Todo bin.")
        self.btn_morph_all_gray = _mini_button("Todo gris")
        mr_layout.addWidget(self.btn_morph_bin)
        mr_layout.addWidget(self.btn_morph_gray)
        mr_layout.addWidget(self.btn_morph_all_bin)
        mr_layout.addWidget(self.btn_morph_all_gray)
        sec_morph.add_widget(mode_row)

        # Operaciones básicas
        lbl_basic = QLabel("  Operaciones básicas")
        lbl_basic.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px; padding: 4px 0 0;")
        sec_morph.add_widget(lbl_basic)
        basic_row = QWidget()
        basic_row.setStyleSheet("background: transparent;")
        br_layout = QHBoxLayout(basic_row)
        br_layout.setContentsMargins(12, 2, 12, 2)
        br_layout.setSpacing(4)
        self.btn_erode   = _mini_button("Erosión")
        self.btn_dilate  = _mini_button("Dilatación")
        self.btn_open    = _mini_button("Apertura")
        self.btn_close   = _mini_button("Cierre")
        br_layout.addWidget(self.btn_erode)
        br_layout.addWidget(self.btn_dilate)
        br_layout.addWidget(self.btn_open)
        br_layout.addWidget(self.btn_close)
        sec_morph.add_widget(basic_row)

        # Morfología Binaria avanzada
        lbl_adv_bin = QLabel("  Morfología Binaria")
        lbl_adv_bin.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px; padding: 4px 0 0;")
        sec_morph.add_widget(lbl_adv_bin)
        adv_bin_row = QWidget()
        adv_bin_row.setStyleSheet("background: transparent;")
        ab_layout = QHBoxLayout(adv_bin_row)
        ab_layout.setContentsMargins(12, 2, 12, 2)
        ab_layout.setSpacing(4)
        self.btn_boundary   = _mini_button("Frontera")
        self.btn_hit_miss   = _mini_button("Hit-Miss")
        self.btn_thin       = _mini_button("Adelgaz.")
        self.btn_skeleton   = _mini_button("Esqueleto")
        ab_layout.addWidget(self.btn_boundary)
        ab_layout.addWidget(self.btn_hit_miss)
        ab_layout.addWidget(self.btn_thin)
        ab_layout.addWidget(self.btn_skeleton)
        sec_morph.add_widget(adv_bin_row)

        # Morfología en Grises (Latticce) avanzada
        lbl_adv_gray = QLabel("  Morfología en Grises")
        lbl_adv_gray.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px; padding: 4px 0 0;")
        sec_morph.add_widget(lbl_adv_gray)
        adv_gray_row = QWidget()
        adv_gray_row.setStyleSheet("background: transparent;")
        ag_layout = QHBoxLayout(adv_gray_row)
        ag_layout.setContentsMargins(12, 2, 12, 2)
        ag_layout.setSpacing(4)
        self.btn_grad_sym  = _mini_button("∇ Sim.")
        self.btn_grad_ero  = _mini_button("∇ Ero.")
        self.btn_grad_dil  = _mini_button("∇ Dil.")
        ag_layout.addWidget(self.btn_grad_sym)
        ag_layout.addWidget(self.btn_grad_ero)
        ag_layout.addWidget(self.btn_grad_dil)
        sec_morph.add_widget(adv_gray_row)

        hat_row = QWidget()
        hat_row.setStyleSheet("background: transparent;")
        ht_layout = QHBoxLayout(hat_row)
        ht_layout.setContentsMargins(12, 2, 12, 6)
        ht_layout.setSpacing(4)
        self.btn_top_hat  = _mini_button("Top Hat")
        self.btn_bot_hat  = _mini_button("Bot Hat")
        self.btn_smooth   = _mini_button("Suavizado")
        ht_layout.addWidget(self.btn_top_hat)
        ht_layout.addWidget(self.btn_bot_hat)
        ht_layout.addWidget(self.btn_smooth)
        sec_morph.add_widget(hat_row)

        inner_layout.addWidget(sec_morph)

        # ── Sección: Análisis ─────────────────────────────────────────
        sec_analysis = SidebarSection("Análisis")
        self.btn_hist = _sidebar_button("  Histograma detallado")
        sec_analysis.add_widget(self.btn_hist)
        inner_layout.addWidget(sec_analysis)

        # ── Histograma en vivo ────────────────────────────────────────
        sec_live = SidebarSection("Histograma en vivo")
        self.live_histogram = LiveHistogramCanvas()
        sec_live.add_widget(self.live_histogram)
        inner_layout.addWidget(sec_live)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        # Footer de estado
        self._status_footer = QWidget()
        self._status_footer.setStyleSheet(
            f"background: {SIDEBAR_BG}; border-top: 1px solid {SIDEBAR_BORDER};"
        )
        sf_layout = QVBoxLayout(self._status_footer)
        sf_layout.setContentsMargins(14, 8, 14, 10)
        sf_layout.setSpacing(2)
        self._lbl_image_name = QLabel("Sin imagen")
        self._lbl_image_name.setStyleSheet(f"color: {SIDEBAR_TEXT}; font-size: 11px; font-weight: 500;")
        self._lbl_image_info = QLabel("")
        self._lbl_image_info.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px;")
        sf_layout.addWidget(self._lbl_image_name)
        sf_layout.addWidget(self._lbl_image_info)
        layout.addWidget(self._status_footer)

        return sidebar

    # ── Área principal ───────────────────────────────────────────────────────

    def _build_main_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar con pestañas + botón comparar
        toolbar = QWidget()
        toolbar.setFixedHeight(38)
        toolbar.setStyleSheet(
            "background: #ffffff; border-bottom: 1px solid #dde1e8;"
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(12, 0, 12, 0)
        tb_layout.setSpacing(0)

        tab_style = """
            QPushButton {{
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                color: #888;
                font-size: 12px;
                padding: 0 16px;
                height: 38px;
            }}
            QPushButton:checked {{
                border-bottom: 2px solid #1e2b3c;
                color: #1e2b3c;
                font-weight: 600;
            }}
            QPushButton:hover:!checked {{ color: #444; }}
        """
        self._tab_original  = QPushButton("Original")
        self._tab_resultado = QPushButton("Resultado")
        self._tab_comparar  = QPushButton("Comparar")
        for tab in (self._tab_original, self._tab_resultado, self._tab_comparar):
            tab.setCheckable(True)
            tab.setStyleSheet(tab_style)
        self._tab_original.setChecked(True)

        self._tab_original.clicked.connect(lambda: self._switch_tab(0))
        self._tab_resultado.clicked.connect(lambda: self._switch_tab(1))
        self._tab_comparar.clicked.connect(lambda: self._switch_tab(2))

        tb_layout.addWidget(self._tab_original)
        tb_layout.addWidget(self._tab_resultado)
        tb_layout.addWidget(self._tab_comparar)
        tb_layout.addStretch()

        # Info dinámica de imagen en toolbar
        self._toolbar_info = QLabel("")
        self._toolbar_info.setStyleSheet("color: #aaa; font-size: 11px;")
        tb_layout.addWidget(self._toolbar_info)

        layout.addWidget(toolbar)

        # Stack de vistas (simulado con QSplitter para comparar)
        self._stack = QWidget()
        self._stack_layout = QVBoxLayout(self._stack)
        self._stack_layout.setContentsMargins(0, 0, 0, 0)

        # Vista 0: Original
        self._view_original = ImageViewer("Carga una imagen para comenzar")

        # Vista 1: Resultado
        self._view_result = ImageViewer("Aplica una transformación para ver el resultado")

        # Vista 2: Comparación lado a lado
        self._compare_widget = QWidget()
        cmp_layout = QHBoxLayout(self._compare_widget)
        cmp_layout.setContentsMargins(0, 0, 0, 0)
        cmp_layout.setSpacing(1)
        self._cmp_original = ImageViewer("Original")
        self._cmp_result   = ImageViewer("Resultado")
        cmp_lbl_style = f"""
            QLabel {{
                font-size: 10px; font-weight: 600; letter-spacing: .05em;
                color: {SIDEBAR_LABEL}; background: {CANVAS_BG};
                padding: 4px 8px;
            }}
        """
        # Envolver cada visor en un contenedor con etiqueta
        for viewer, label_text in [(self._cmp_original, "ORIGINAL"),
                                    (self._cmp_result,   "RESULTADO")]:
            wrap = QWidget()
            wrap.setStyleSheet(f"background: {CANVAS_BG};")
            wl = QVBoxLayout(wrap)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(0)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(cmp_lbl_style)
            wl.addWidget(lbl)
            wl.addWidget(viewer, stretch=1)
            cmp_layout.addWidget(wrap, stretch=1)

        # Agregar las tres vistas, solo mostrar la activa
        self._stack_layout.addWidget(self._view_original)
        self._stack_layout.addWidget(self._view_result)
        self._stack_layout.addWidget(self._compare_widget)
        self._view_result.hide()
        self._compare_widget.hide()

        layout.addWidget(self._stack, stretch=1)

        # Barra de estado inferior
        status_bar = QWidget()
        status_bar.setFixedHeight(24)
        status_bar.setStyleSheet(
            f"background: {STATUS_BG}; border-top: 1px solid #dde1e8;"
        )
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(12, 0, 12, 0)
        self._status_label = QLabel("Listo · Carga una imagen para comenzar")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        self._version_label = QLabel("v3.0")
        self._version_label.setStyleSheet(
            "color: #aaa; font-size: 10px; padding: 0 6px;"
            "background: #e4e8f0; border-radius: 3px;"
        )
        sb_layout.addWidget(self._status_label)
        sb_layout.addStretch()
        sb_layout.addWidget(self._version_label)
        layout.addWidget(status_bar)

        return container

    # ── Navegación de pestañas ───────────────────────────────────────────────

    def _switch_tab(self, index: int):
        """Muestra la vista correspondiente y actualiza el estado de las pestañas."""
        for i, tab in enumerate((self._tab_original, self._tab_resultado, self._tab_comparar)):
            tab.setChecked(i == index)

        self._view_original.setVisible(index == 0)
        self._view_result.setVisible(index == 1)
        self._compare_widget.setVisible(index == 2)

        # Exponer índice para que el controlador sepa cuál está activa
        self._current_tab = index

    # ── Propiedad tabs (compatibilidad con controller.py) ────────────────────
    # El controlador usa self.view.tabs.currentIndex() — simulamos esa API.

    class _TabsProxy:
        """Proxy para mantener compatibilidad con el controller existente."""
        def __init__(self, window):
            self._w = window
        def currentIndex(self) -> int:
            return getattr(self._w, "_current_tab", 0)

    @property
    def tabs(self):
        return self._TabsProxy(self)

    # ── API pública ──────────────────────────────────────────────────────────

    def show_original(self, img):
        """Muestra la imagen original en su visor y en el panel de comparación."""
        self._view_original.set_image(img)
        self._cmp_original.set_image(img)
        if img is not None:
            h, w = img.shape[:2]
            self._lbl_image_name.setText(
                getattr(self, "_current_image_name", "imagen")
            )
            self._lbl_image_info.setText(f"{w} × {h} px · RGB")
            self._toolbar_info.setText(f"{w} × {h}")

    def show_result(self, img):
        """Muestra el resultado y activa la pestaña resultado."""
        self._view_result.set_image(img)
        self._cmp_result.set_image(img)
        self._switch_tab(1)

    def show_status(self, msg: str):
        """Actualiza la barra de estado inferior."""
        self._status_label.setText(msg)
        self.btn_save.setEnabled(True)

    def update_live_histogram(self, hists, mode: str = "rgb"):
        """Actualiza el histograma en vivo del sidebar."""
        self.live_histogram.update_histogram(hists, mode)

    def set_image_name(self, name: str):
        """Guarda el nombre de imagen para mostrarlo en el footer."""
        self._current_image_name = name

    # ── Señales ──────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.btn_load.clicked.connect(self.controller.load_image)
        self.btn_save.clicked.connect(self.controller.save_result)
        self.btn_rgb.clicked.connect(self.controller.show_rgb_components)
        self.btn_gray.clicked.connect(self.controller.convert_to_gray)
        self.btn_apply_map.clicked.connect(self.controller.apply_map)
        self.btn_apply_model.clicked.connect(self.controller.apply_color_model)
        self.btn_binary_fixed.clicked.connect(
            lambda: self.controller.apply_binary("fixed")
        )
        self.btn_binary_otsu.clicked.connect(
            lambda: self.controller.apply_binary("otsu")
        )
        self.btn_binary_adaptive.clicked.connect(
            lambda: self.controller.apply_binary("adaptive")
        )
        self.btn_hist.clicked.connect(self.controller.show_histogram)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(str(v))
        )

        # ── 3-a Ruido ─────────────────────────────────────────────────
        self.btn_apply_noise.clicked.connect(self.controller.apply_noise)
        self.btn_filter_median.clicked.connect(
            lambda: self.controller.apply_filter("median")
        )
        self.btn_filter_gauss.clicked.connect(
            lambda: self.controller.apply_filter("gaussian")
        )

        # ── 3-b Operaciones ───────────────────────────────────────────
        self.btn_arith_add.clicked.connect(
            lambda: self.controller.apply_arithmetic("add")
        )
        self.btn_arith_sub.clicked.connect(
            lambda: self.controller.apply_arithmetic("subtract")
        )
        self.btn_arith_mul.clicked.connect(
            lambda: self.controller.apply_arithmetic("multiply")
        )
        self.btn_logic_and.clicked.connect(
            lambda: self.controller.apply_logic("and")
        )
        self.btn_logic_or.clicked.connect(
            lambda: self.controller.apply_logic("or")
        )
        self.btn_logic_xor.clicked.connect(
            lambda: self.controller.apply_logic("xor")
        )
        self.btn_logic_not.clicked.connect(
            lambda: self.controller.apply_logic("not")
        )
        self.btn_rel_gt.clicked.connect(
            lambda: self.controller.apply_relational("gt")
        )
        self.btn_rel_lt.clicked.connect(
            lambda: self.controller.apply_relational("lt")
        )
        self.btn_rel_eq.clicked.connect(
            lambda: self.controller.apply_relational("eq")
        )

        # ── 3-c Componentes conexas ───────────────────────────────────
        self.btn_cc_v4.clicked.connect(
            lambda: self.controller.apply_connected_components(4)
        )
        self.btn_cc_v8.clicked.connect(
            lambda: self.controller.apply_connected_components(8)
        )
        self.btn_cc_cmp.clicked.connect(
            self.controller.compare_connected_components
        )

        # ── 4 Morfología Matemática ───────────────────────────────────
        self.btn_morph_bin.clicked.connect(
            lambda: self.controller.apply_morph_single("bin")
        )
        self.btn_morph_gray.clicked.connect(
            lambda: self.controller.apply_morph_single("gray")
        )
        self.btn_morph_all_bin.clicked.connect(
            lambda: self.controller.apply_morph_all("bin")
        )
        self.btn_morph_all_gray.clicked.connect(
            lambda: self.controller.apply_morph_all("gray")
        )
        self.btn_erode.clicked.connect(
            lambda: self.controller.apply_morph_op("erosion")
        )
        self.btn_dilate.clicked.connect(
            lambda: self.controller.apply_morph_op("dilation")
        )
        self.btn_open.clicked.connect(
            lambda: self.controller.apply_morph_op("opening")
        )
        self.btn_close.clicked.connect(
            lambda: self.controller.apply_morph_op("closing")
        )
        self.btn_boundary.clicked.connect(
            lambda: self.controller.apply_morph_op("boundary")
        )
        self.btn_hit_miss.clicked.connect(
            lambda: self.controller.apply_morph_op("hit_or_miss")
        )
        self.btn_thin.clicked.connect(
            lambda: self.controller.apply_morph_op("thinning")
        )
        self.btn_skeleton.clicked.connect(
            lambda: self.controller.apply_morph_op("skeleton")
        )
        self.btn_grad_sym.clicked.connect(
            lambda: self.controller.apply_morph_op("grad_symmetric")
        )
        self.btn_grad_ero.clicked.connect(
            lambda: self.controller.apply_morph_op("grad_erosion")
        )
        self.btn_grad_dil.clicked.connect(
            lambda: self.controller.apply_morph_op("grad_dilation")
        )
        self.btn_top_hat.clicked.connect(
            lambda: self.controller.apply_morph_op("top_hat")
        )
        self.btn_bot_hat.clicked.connect(
            lambda: self.controller.apply_morph_op("bot_hat")
        )
        self.btn_smooth.clicked.connect(
            lambda: self.controller.apply_morph_op("smooth")
        )
