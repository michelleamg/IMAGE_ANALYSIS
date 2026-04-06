"""Ventana de histograma detallado — versión 3.0.
Reemplaza las figuras flotantes de matplotlib por una ventana Qt dedicada con:
  ✓ Gráfica grande que ocupa la mayor parte del espacio
  ✓ Imagen de referencia pequeña en esquina superior izquierda
  ✓ Tabla de estadísticas con tarjetas visuales por canal
  ✓ Estilo coherente con el resto de la app (fondo oscuro)
  ✓ Pestaña por canal en modo RGB / mapa de color
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QTabWidget, QFrame, QSizePolicy, QGridLayout, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as mticker

# ── Paleta ────────────────────────────────────────────────────────────────────
BG       = "#1e2b3c"
BG_DARK  = "#141e2b"
BG_CARD  = "#243447"
BORDER   = "#2a3f57"
TEXT     = "#a8b8cc"
TEXT_HI  = "#e8f0f8"
LABEL    = "#5a7a99"
ACCENT   = "#4a9eff"

CHAN_COLORS = {
    "r": ("#ff6b6b", "#3a0000"),
    "g": ("#6bffa0", "#003a10"),
    "b": ("#6bb5ff", "#001030"),
    "gray": ("#c8d8e8", "#1a2030"),
    "0":  ("#222222", "#111111"),
    "255":("#f0f0f0", "#2a2a2a"),
}


def _np_to_pixmap(img: np.ndarray, w: int, h: int) -> QPixmap:
    ih, iw = img.shape[:2]
    if img.ndim == 2:
        q = QImage(img.tobytes(), iw, ih, iw, QImage.Format_Grayscale8)
    else:
        q = QImage(img.tobytes(), iw, ih, 3 * iw, QImage.Format_RGB888)
    return QPixmap.fromImage(q).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ── Canvas de matplotlib estilizado ──────────────────────────────────────────
class _HistCanvas(FigureCanvas):
    def __init__(self, width=9, height=4.2):
        # tight_layout=True eliminado: es incompatible con fig.clear()+add_subplot
        # repetido en cada redibujado y provoca el UserWarning que congela la UI.
        self.fig = Figure(figsize=(width, height), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {BG_DARK};")

    def _styled_ax(self, ax):
        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=LABEL, labelsize=8)
        ax.xaxis.label.set_color(LABEL)
        ax.yaxis.label.set_color(LABEL)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
        )
        return ax

    def _apply_margins(self):
        """Márgenes fijos seguros, sin llamar a tight_layout."""
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.11)

    def plot_rgb(self, hists: dict, title: str = ""):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._styled_ax(ax)

        colors = {"r": ("#ff4444", "Rojo"), "g": ("#44ff88", "Verde"), "b": ("#44aaff", "Azul")}
        x = np.arange(256)
        for key, (color, label) in colors.items():
            if key in hists:
                ax.fill_between(x, hists[key], alpha=0.25, color=color)
                ax.plot(x, hists[key], color=color, linewidth=1.4,
                        alpha=0.9, label=label)

        ax.set_xlim(0, 255)
        ax.set_xlabel("Nivel de intensidad", fontsize=9, color=LABEL)
        ax.set_ylabel("Frecuencia", fontsize=9, color=LABEL)
        ax.grid(True, alpha=0.12, color=BORDER, linewidth=0.6)
        ax.legend(fontsize=9, facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.9)
        if title:
            ax.set_title(title, fontsize=10, color=TEXT, pad=8)
        self._apply_margins()
        self.draw()

    def plot_single(self, hist: np.ndarray, color: str, fill_color: str,
                    stats: dict, channel_label: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._styled_ax(ax)

        x = np.arange(256)
        ax.fill_between(x, hist, alpha=0.22, color=color)
        ax.plot(x, hist, color=color, linewidth=1.6)

        if stats:
            for val, ls, lbl in [
                (stats["media"],   "--", f"Media {stats['media']:.1f}"),
                (stats["mediana"], ":" , f"Mediana {stats['mediana']}"),
                (stats["moda"],    "-.", f"Moda {stats['moda']}"),
            ]:
                ax.axvline(val, color=color, linestyle=ls, linewidth=1.2,
                           alpha=0.8, label=lbl)

        # Región IQR sombreada
        if stats:
            ax.axvspan(stats["percentil_25"], stats["percentil_75"],
                       alpha=0.08, color=color, label="IQR (P25–P75)")

        ax.set_xlim(0, 255)
        ax.set_xlabel("Nivel de intensidad", fontsize=9, color=LABEL)
        ax.set_ylabel("Frecuencia", fontsize=9, color=LABEL)
        ax.grid(True, alpha=0.12, color=BORDER, linewidth=0.6)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.9)
        ax.set_title(f"Canal {channel_label}", fontsize=10, color=TEXT, pad=8)
        self._apply_margins()
        self.draw()

    def plot_gray(self, hist: np.ndarray, stats: dict):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._styled_ax(ax)

        x = np.arange(256)
        ax.fill_between(x, hist, alpha=0.25, color="#c8d8e8")
        ax.plot(x, hist, color="#c8d8e8", linewidth=1.6)

        if stats:
            styles = [
                (stats["media"],   "#ff6b6b", "--", f"Media {stats['media']:.1f}"),
                (stats["mediana"], "#6bffa0", ":" , f"Mediana {stats['mediana']}"),
                (stats["moda"],    "#6bb5ff", "-.", f"Moda {stats['moda']}"),
            ]
            for val, col, ls, lbl in styles:
                ax.axvline(val, color=col, linestyle=ls, linewidth=1.4,
                           alpha=0.9, label=lbl)
            ax.axvspan(stats["percentil_25"], stats["percentil_75"],
                       alpha=0.08, color="#c8d8e8", label="IQR (P25–P75)")

        ax.set_xlim(0, 255)
        ax.set_xlabel("Nivel de gris", fontsize=9, color=LABEL)
        ax.set_ylabel("Frecuencia",    fontsize=9, color=LABEL)
        ax.grid(True, alpha=0.12, color=BORDER, linewidth=0.6)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.9)
        ax.set_title("Histograma de intensidades", fontsize=10, color=TEXT, pad=8)
        self._apply_margins()
        self.draw()

    def plot_binary(self, hist: np.ndarray):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._styled_ax(ax)

        vals   = [hist[0], hist[255]]
        labels = ["Negro (0)", "Blanco (255)"]
        colors = ["#444455", "#e8e8f0"]
        bars   = ax.bar([0, 1], vals, color=colors, edgecolor=BORDER,
                        width=0.5, zorder=3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{int(val):,}".replace(",", " "),
                    ha="center", va="bottom", fontsize=9, color=TEXT)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Píxeles", fontsize=9, color=LABEL)
        ax.grid(True, axis="y", alpha=0.12, color=BORDER, linewidth=0.6)
        ax.set_title("Distribución binaria", fontsize=10, color=TEXT, pad=8)
        self._apply_margins()
        self.draw()


# ── Tarjeta de estadísticas ───────────────────────────────────────────────────
class _StatCard(QWidget):
    """Muestra un único valor estadístico con etiqueta y acento de color."""

    def __init__(self, label: str, value: str, accent: str = ACCENT):
        super().__init__()
        self.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER};"
            f"border-radius: 6px; border-left: 3px solid {accent};"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {LABEL}; font-size: 10px; background: transparent; border: none;")
        val = QLabel(value)
        val.setStyleSheet(f"color: {TEXT_HI}; font-size: 13px; font-weight: 600; background: transparent; border: none;")
        layout.addWidget(lbl)
        layout.addWidget(val)


def _stats_grid(stats: dict, accent: str) -> QWidget:
    """Construye el grid de tarjetas de estadísticas para un canal."""
    if stats is None:
        lbl = QLabel("Sin datos")
        lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px;")
        return lbl

    widget = QWidget()
    widget.setStyleSheet(f"background: {BG};")
    grid = QGridLayout(widget)
    grid.setContentsMargins(0, 6, 0, 6)
    grid.setSpacing(6)

    entries = [
        ("Media",          f"{stats['media']:.2f}"),
        ("Mediana",        f"{stats['mediana']}"),
        ("Moda",           f"{stats['moda']}"),
        ("Desv. estándar", f"{stats['desviacion']:.2f}"),
        ("Varianza",       f"{stats['varianza']:.1f}"),
        ("Asimetría",      f"{stats['asimetria']:.3f}"),
        ("Curtosis",       f"{stats['curtosis']:.3f}"),
        ("Entropía",       f"{stats['entropia']:.3f} bits"),
        ("Energía",        f"{stats['energia']:.5f}"),
        ("Rango",          f"[{stats['min']}, {stats['max']}]"),
        ("Percentil 25",   f"{stats['percentil_25']}"),
        ("Percentil 75",   f"{stats['percentil_75']}"),
    ]
    cols = 3
    for i, (lbl, val) in enumerate(entries):
        grid.addWidget(_StatCard(lbl, val, accent), i // cols, i % cols)
    for c in range(cols):
        grid.setColumnStretch(c, 1)
    return widget


# ── Ventana principal ─────────────────────────────────────────────────────────
class HistogramWindow(QMainWindow):
    """Ventana dedicada al histograma detallado."""

    def __init__(self, mode: str, image_name: str,
                 img=None, hists=None, stats_data=None):
        """
        mode        : "original" | "gray" | "colormap" | "binary"
        image_name  : nombre para el título
        img         : array numpy RGB o gris (imagen de referencia)
        hists       : dict {"r","g","b"} o array numpy (gray/binary)
        stats_data  : dict por canal {"r": stats, "g": stats, "b": stats}
                      o stats único para gray/binary
        """
        super().__init__()
        self.setWindowTitle(f"Histograma — {image_name}")
        self.setMinimumSize(1000, 640)
        self.resize(1200, 740)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Panel izquierdo: imagen de referencia + stats globales
        root.addWidget(self._build_left(img, mode, hists, stats_data))

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {BORDER};")
        root.addWidget(sep)

        # Panel derecho: gráfica(s) + tabla de estadísticas
        root.addWidget(self._build_right(mode, hists, stats_data, image_name), stretch=1)

        self.statusBar().setStyleSheet(
            f"background: {BG}; color: {LABEL}; font-size: 11px;"
        )
        self.statusBar().showMessage(f"Imagen: {image_name}  ·  Modo: {mode}")

    # ── Panel izquierdo ───────────────────────────────────────────────────────

    def _build_left(self, img, mode, hists, stats_data) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(230)
        panel.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Título
        hdr = QLabel("  Referencia")
        hdr.setFixedHeight(40)
        hdr.setStyleSheet(
            f"background: {BG_CARD}; color: {TEXT_HI}; font-size: 13px;"
            f"font-weight: 600; border-bottom: 2px solid {ACCENT};"
        )
        layout.addWidget(hdr)

        # Miniatura de la imagen
        if img is not None:
            thumb = QLabel()
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setFixedHeight(200)
            thumb.setStyleSheet(f"background: {BG_DARK};")
            px = _np_to_pixmap(img, 220, 196)
            thumb.setPixmap(px)
            layout.addWidget(thumb)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {BORDER}; max-height: 1px;")
        layout.addWidget(sep)

        # Resumen rápido del modo
        mode_labels = {
            "original": ("RGB", ACCENT),
            "gray":     ("Grises", "#c8d8e8"),
            "colormap": ("Mapa de color", "#cc88ff"),
            "binary":   ("Binaria", "#ffaa44"),
        }
        mode_text, mode_color = mode_labels.get(mode, ("—", LABEL))

        mode_lbl = QLabel(f"  Tipo:  {mode_text}")
        mode_lbl.setStyleSheet(
            f"color: {mode_color}; font-size: 11px; font-weight: 600;"
            f"padding: 8px 0 4px; background: transparent;"
        )
        layout.addWidget(mode_lbl)

        # Info rápida si hay stats
        if isinstance(stats_data, dict) and "media" in stats_data:
            # Stats de un solo canal (gray/binary)
            for lbl_txt, val_txt in [
                ("Media",    f"{stats_data['media']:.1f}"),
                ("Mediana",  f"{stats_data['mediana']}"),
                ("Entropía", f"{stats_data['entropia']:.2f} bits"),
                ("Rango",    f"[{stats_data['min']}, {stats_data['max']}]"),
            ]:
                row = QWidget()
                row.setStyleSheet("background: transparent;")
                rl = QHBoxLayout(row)
                rl.setContentsMargins(12, 3, 12, 3)
                l1 = QLabel(lbl_txt)
                l1.setStyleSheet(f"color: {LABEL}; font-size: 11px; background: transparent;")
                l2 = QLabel(val_txt)
                l2.setStyleSheet(f"color: {TEXT_HI}; font-size: 11px; font-weight: 600; background: transparent;")
                l2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                rl.addWidget(l1)
                rl.addWidget(l2)
                layout.addWidget(row)

        layout.addStretch()
        return panel

    # ── Panel derecho ─────────────────────────────────────────────────────────

    def _build_right(self, mode, hists, stats_data, image_name) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if mode in ("original", "colormap"):
            layout.addWidget(self._build_rgb_tabs(hists, stats_data, mode, image_name))
        elif mode == "gray":
            layout.addWidget(self._build_single_view(hists, stats_data, "gray"))
        elif mode == "binary":
            layout.addWidget(self._build_binary_view(hists, stats_data))

        return panel

    # ── Vista RGB con pestañas por canal ─────────────────────────────────────

    def _build_rgb_tabs(self, hists, stats_data, mode, image_name) -> QWidget:
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {BG}; }}
            QTabBar::tab {{
                background: {BG}; color: {LABEL};
                padding: 8px 20px; font-size: 12px; border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {TEXT_HI}; background: {BG_CARD};
                border-bottom: 2px solid {ACCENT};
            }}
            QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
        """)

        # Pestaña general RGB superpuesto
        tabs.addTab(
            self._build_overview_tab(hists, image_name),
            "RGB superpuesto"
        )

        # Pestaña por canal
        chan_cfg = [
            ("r", "Rojo",  "#ff4444"),
            ("g", "Verde", "#44ff88"),
            ("b", "Azul",  "#44aaff"),
        ]
        for key, label, color in chan_cfg:
            if key in hists:
                s = stats_data.get(key) if isinstance(stats_data, dict) else None
                tabs.addTab(
                    self._build_single_view(hists[key], s, key, color, label),
                    label
                )
        return tabs

    def _build_overview_tab(self, hists, image_name) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        canvas = _HistCanvas(width=9, height=5)
        canvas.plot_rgb(hists, f"Histograma RGB — {image_name}")
        layout.addWidget(canvas, stretch=1)
        return widget

    # ── Vista de canal individual ─────────────────────────────────────────────

    def _build_single_view(self, hist, stats, chan_key,
                           color="#c8d8e8", label="Intensidad") -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(10)

        # Gráfica grande
        canvas = _HistCanvas(width=9, height=4.8)
        if chan_key == "gray":
            canvas.plot_gray(hist, stats)
        else:
            canvas.plot_single(hist, color, color, stats, label)
        layout.addWidget(canvas, stretch=1)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {BORDER}; max-height: 1px;")
        layout.addWidget(sep)

        # Grid de tarjetas de estadísticas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedHeight(160)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG}; border: none; }}
            QScrollBar:horizontal {{
                background: {BG}; height: 4px; border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {BORDER}; border-radius: 2px;
            }}
        """)
        scroll.setWidget(_stats_grid(stats, color))
        layout.addWidget(scroll)

        return widget

    # ── Vista binaria ─────────────────────────────────────────────────────────

    def _build_binary_view(self, hist, stats) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(10)

        canvas = _HistCanvas(width=9, height=4.8)
        canvas.plot_binary(hist)
        layout.addWidget(canvas, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {BORDER}; max-height: 1px;")
        layout.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedHeight(160)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {BG}; border: none; }}"
        )
        scroll.setWidget(_stats_grid(stats, "#ffaa44"))
        layout.addWidget(scroll)

        return widget
