"""Ventana para mostrar modelos de color — versión 3.0.
- Cambios:
    ✓ Imágenes de canal más grandes (se adaptan al tamaño de ventana)
    ✓ Vista de comparación original vs modelo lado a lado mejorada
    ✓ Guardado de canal individual en PNG con diálogo de selección
    ✓ Barra de estado con info del canal activo
    ✓ Estilo coherente con el sidebar azul oscuro de la app
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QTabWidget, QPushButton, QFileDialog, QScrollArea,
    QFrame, QSizePolicy, QGridLayout, QSplitter,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

# ── Paleta (igual que view.py para consistencia visual) ───────────────────────
SIDEBAR_BG      = "#1e2b3c"
SIDEBAR_BORDER  = "#2a3f57"
SIDEBAR_TEXT    = "#a8b8cc"
SIDEBAR_TEXT_HI = "#e8f0f8"
SIDEBAR_LABEL   = "#5a7a99"
SIDEBAR_ACTIVE  = "#2d4a6b"
ACCENT          = "#4a9eff"
CANVAS_BG       = "#1a1a2e"


def _make_pixmap(img_rgb: np.ndarray, target_size: QSize) -> QPixmap:
    """Convierte array RGB/gris a QPixmap escalado manteniendo aspecto."""
    if img_rgb is None:
        return QPixmap()
    h, w = img_rgb.shape[:2]
    if img_rgb.ndim == 2:
        q = QImage(img_rgb.tobytes(), w, h, w, QImage.Format_Grayscale8)
    else:
        q = QImage(img_rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q).scaled(
        target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


# ── Visor redimensionable ─────────────────────────────────────────────────────
class _ResizableViewer(QLabel):
    def __init__(self, img_rgb: np.ndarray):
        super().__init__()
        self._img = img_rgb
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {CANVAS_BG};")

    def _refresh(self):
        if self._img is not None and self.width() > 10:
            self.setPixmap(_make_pixmap(
                self._img, QSize(self.width() - 4, self.height() - 4)
            ))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def showEvent(self, e):
        super().showEvent(e)
        self._refresh()


# ── Tarjeta de canal individual ───────────────────────────────────────────────
class ChannelCard(QWidget):
    """Tarjeta con imagen, stats y botón de guardado individual en PNG."""

    _ACCENTS = {
        "R": "#ff6b6b", "G": "#6bffa0", "B": "#6bb5ff",
        "C": "#00e5ff", "M": "#ff6bcc", "Y": "#ffe066", "K": "#c0c0c0",
        "H": "#ffaa44", "S": "#44ffdd", "V": "#cc88ff",
        "I": "#ff88bb", "U": "#88ccff",
        "L": "#ffffff",  "A": "#ff9999",
        "X": "#ffcc99", "Z": "#99ccff",
    }

    def __init__(self, raw_data: np.ndarray, channel_name: str,
                 display_img: np.ndarray, parent=None):
        super().__init__(parent)
        self.channel_name = channel_name.upper()
        self.display_img  = display_img
        self.raw_data     = raw_data
        accent = self._ACCENTS.get(self.channel_name, "#aaaaaa")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Cabecera
        hdr = QWidget()
        hdr.setFixedHeight(30)
        hdr.setStyleSheet(
            f"background: {SIDEBAR_BG}; border-bottom: 2px solid {accent};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 0, 8, 0)
        lbl = QLabel(f"Canal  {self.channel_name}")
        lbl.setStyleSheet(
            f"color: {accent}; font-size: 12px; font-weight: 600;"
        )
        stats = QLabel(self._stats(raw_data))
        stats.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px;")
        hl.addWidget(lbl)
        hl.addStretch()
        hl.addWidget(stats)
        layout.addWidget(hdr)

        # Imagen
        self.viewer = _ResizableViewer(display_img)
        layout.addWidget(self.viewer, stretch=1)

        # Footer con botón guardar
        ftr = QWidget()
        ftr.setFixedHeight(34)
        ftr.setStyleSheet(f"background: {SIDEBAR_BG};")
        fl = QHBoxLayout(ftr)
        fl.setContentsMargins(8, 4, 8, 4)
        btn = QPushButton(f"  Guardar {self.channel_name}  como PNG")
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {SIDEBAR_ACTIVE}; color: {SIDEBAR_TEXT};
                border: 1px solid {SIDEBAR_BORDER}; border-radius: 4px;
                font-size: 11px; padding: 3px 10px;
            }}
            QPushButton:hover {{
                background: {accent}; color: #000;
                border-color: {accent};
            }}
            QPushButton:pressed {{ opacity: 0.8; }}
        """)
        btn.clicked.connect(self._save)
        fl.addWidget(btn)
        layout.addWidget(ftr)

        self.setStyleSheet(
            f"border: 1px solid {SIDEBAR_BORDER}; border-radius: 6px;"
            f"background: {CANVAS_BG};"
        )

    @staticmethod
    def _stats(data: np.ndarray) -> str:
        if data is None:
            return ""
        return f"min {int(data.min())}  max {int(data.max())}  μ {data.mean():.1f}"

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, f"Guardar canal {self.channel_name}",
            f"canal_{self.channel_name.lower()}.png",
            "PNG (*.png)",
        )
        if not path:
            return
        data = self.raw_data
        if data.dtype != np.uint8:
            data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(path, data)


# ── Ventana principal ─────────────────────────────────────────────────────────
class ColorModelWindow(QMainWindow):

    def __init__(self, model_data: dict, model_name: str,
                 original_image: np.ndarray):
        super().__init__()
        self.model_data     = model_data
        self.model_name     = model_name
        self.original_image = original_image

        self.setWindowTitle(f"Modelo de color — {model_name}")
        self.setMinimumSize(1100, 680)
        self.resize(1300, 760)
        self.setStyleSheet(f"QMainWindow {{ background: {SIDEBAR_BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_info_panel())
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {SIDEBAR_BORDER};")
        root.addWidget(sep)
        root.addWidget(self._build_main_panel(), stretch=1)

        channels = [k for k in model_data if k != "combined"]
        h, w = model_data["combined"].shape[:2]
        self.statusBar().setStyleSheet(
            f"background:{SIDEBAR_BG}; color:{SIDEBAR_LABEL}; font-size:11px;"
        )
        self.statusBar().showMessage(
            f"Modelo {model_name}  ·  {len(channels)} canales  ·  {w}×{h} px"
        )

    # ── Panel izquierdo ───────────────────────────────────────────────────────

    def _build_info_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(220)
        panel.setStyleSheet(f"background: {SIDEBAR_BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Título del modelo
        title = QLabel(f"  {self.model_name}")
        title.setFixedHeight(52)
        title.setStyleSheet(
            f"background: {SIDEBAR_ACTIVE}; color: {SIDEBAR_TEXT_HI};"
            f"font-size: 22px; font-weight: 700;"
            f"border-bottom: 2px solid {ACCENT};"
        )
        layout.addWidget(title)

        # Descripción
        desc = QLabel(self._get_description())
        desc.setWordWrap(True)
        desc.setStyleSheet(
            f"color: {SIDEBAR_TEXT}; font-size: 11px;"
            f"padding: 12px 14px 10px; line-height: 160%;"
        )
        layout.addWidget(desc)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {SIDEBAR_BORDER}; max-height: 1px; margin: 0 14px;")
        layout.addWidget(sep)

        # Barras de media por canal
        channels = [k for k in self.model_data if k != "combined"]
        for ch in channels:
            data   = self.model_data[ch]
            accent = ChannelCard._ACCENTS.get(ch.upper(), "#aaa")
            mean   = float(np.mean(data)) if data is not None else 0

            row = QWidget()
            row.setStyleSheet("background: transparent;")
            rl  = QHBoxLayout(row)
            rl.setContentsMargins(14, 4, 14, 4)
            rl.setSpacing(8)

            name_l = QLabel(ch.upper())
            name_l.setFixedWidth(22)
            name_l.setStyleSheet(
                f"color: {accent}; font-size: 11px; font-weight: 700;"
            )

            bar_bg = QWidget()
            bar_bg.setFixedHeight(4)
            bar_bg.setStyleSheet(
                f"background: {SIDEBAR_BORDER}; border-radius: 2px;"
            )
            bar_pct = max(4, int(140 * mean / 255))
            bar_fg  = QWidget(bar_bg)
            bar_fg.setFixedSize(bar_pct, 4)
            bar_fg.setStyleSheet(f"background: {accent}; border-radius: 2px;")

            val_l = QLabel(f"{mean:.0f}")
            val_l.setFixedWidth(32)
            val_l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_l.setStyleSheet(f"color: {SIDEBAR_LABEL}; font-size: 10px;")

            rl.addWidget(name_l)
            rl.addWidget(bar_bg, stretch=1)
            rl.addWidget(val_l)
            layout.addWidget(row)

        layout.addStretch()

        btn = QPushButton("  Guardar modelo completo")
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT}; color: #fff; border: none;
                border-radius: 5px; font-size: 12px;
                padding: 8px 14px; margin: 10px 14px 14px;
            }}
            QPushButton:hover {{ background: #3a8eef; }}
            QPushButton:pressed {{ background: #2a7edf; }}
        """)
        btn.clicked.connect(self._save_combined)
        layout.addWidget(btn)
        return panel

    # ── Panel principal ───────────────────────────────────────────────────────

    def _build_main_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {CANVAS_BG}; }}
            QTabBar::tab {{
                background: {SIDEBAR_BG}; color: {SIDEBAR_LABEL};
                padding: 8px 22px; font-size: 12px; border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {SIDEBAR_TEXT_HI}; background: {SIDEBAR_ACTIVE};
                border-bottom: 2px solid {ACCENT};
            }}
            QTabBar::tab:hover:!selected {{ color: {SIDEBAR_TEXT}; }}
        """)

        tabs.addTab(self._build_channels_tab(), "Canales")
        tabs.addTab(self._build_compare_tab(),  "Comparación")
        layout.addWidget(tabs)
        return panel

    # ── Tab canales ───────────────────────────────────────────────────────────

    def _build_channels_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(f"background: {CANVAS_BG};")
        channels = [k for k in self.model_data if k != "combined"]
        n    = len(channels)
        cols = min(n, 4)

        grid = QGridLayout(widget)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setSpacing(10)

        for i, ch in enumerate(channels):
            display = self._prepare_channel(self.model_data[ch], ch)
            card    = ChannelCard(self.model_data[ch], ch, display)
            grid.addWidget(card, i // cols, i % cols)

        for c in range(cols):
            grid.setColumnStretch(c, 1)
        for r in range((n + cols - 1) // cols):
            grid.setRowStretch(r, 1)

        return widget

    # ── Tab comparación ───────────────────────────────────────────────────────

    def _build_compare_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {SIDEBAR_BORDER}; width: 3px; }}"
        )
        splitter.setHandleWidth(3)

        pairs = [
            (self.original_image, "Original"),
            (self._prepare_combined(), f"Modelo  {self.model_name}"),
        ]
        for img, label in pairs:
            pane = QWidget()
            pane.setStyleSheet(f"background: {CANVAS_BG};")
            pl = QVBoxLayout(pane)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.setSpacing(0)

            hdr = QLabel(f"  {label}")
            hdr.setFixedHeight(32)
            hdr.setStyleSheet(
                f"background: {SIDEBAR_ACTIVE}; color: {SIDEBAR_TEXT_HI};"
                f"font-size: 12px; font-weight: 600;"
                f"border-bottom: 1px solid {SIDEBAR_BORDER};"
            )
            pl.addWidget(hdr)
            pl.addWidget(_ResizableViewer(img), stretch=1)

            if img is not None:
                h, w = img.shape[:2]
                info = QLabel(f"  {w} × {h} px")
                info.setFixedHeight(22)
                info.setStyleSheet(
                    f"background:{SIDEBAR_BG}; color:{SIDEBAR_LABEL}; font-size:10px;"
                )
                pl.addWidget(info)

            splitter.addWidget(pane)

        splitter.setSizes([600, 600])
        return splitter

    # ── Preparación de imágenes ───────────────────────────────────────────────

    def _prepare_channel(self, data: np.ndarray, ch: str) -> np.ndarray:
        if data is None:
            return None
        if data.dtype != np.uint8:
            norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            norm = data.copy()
        h, w = norm.shape
        rgb  = np.zeros((h, w, 3), dtype=np.uint8)
        c    = ch.lower()
        if   c == "r":                 rgb[:, :, 0] = norm
        elif c == "g":                 rgb[:, :, 1] = norm
        elif c == "b":                 rgb[:, :, 2] = norm
        elif c == "c":                 rgb[:, :, 1] = norm; rgb[:, :, 2] = norm
        elif c == "m":                 rgb[:, :, 0] = norm; rgb[:, :, 2] = norm
        elif c == "y":                 rgb[:, :, 0] = norm; rgb[:, :, 1] = norm
        elif c in ("k", "l", "v", "i", "s", "u", "x", "z", "a"):
            rgb[:, :, 0] = norm; rgb[:, :, 1] = norm; rgb[:, :, 2] = norm
        elif c == "h":
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_HSV)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        else:
            rgb[:, :, 0] = norm; rgb[:, :, 1] = norm; rgb[:, :, 2] = norm
        return rgb

    def _prepare_combined(self) -> np.ndarray:
        c = self.model_data["combined"]
        cvt = {
            "HSV": cv2.COLOR_HSV2RGB, "HSL": cv2.COLOR_HLS2RGB,
            "LAB": cv2.COLOR_LAB2RGB, "YUV": cv2.COLOR_YUV2RGB,
            "XYZ": cv2.COLOR_XYZ2RGB,
        }
        if self.model_name in cvt:
            return cv2.cvtColor(c, cvt[self.model_name])
        if self.model_name == "CMYK":
            k  = np.clip(self.model_data["k"].astype(np.float32) / 255, 0, 1)
            r  = np.clip((1 - self.model_data["c"].astype(np.float32) / 255) * (1 - k), 0, 1) * 255
            g  = np.clip((1 - self.model_data["m"].astype(np.float32) / 255) * (1 - k), 0, 1) * 255
            b  = np.clip((1 - self.model_data["y"].astype(np.float32) / 255) * (1 - k), 0, 1) * 255
            return cv2.merge([r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)])
        return c  # RGB / HSI ya están en formato correcto

    # ── Guardar modelo completo ───────────────────────────────────────────────

    def _save_combined(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar modelo completo",
            f"modelo_{self.model_name.lower()}.png",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if path:
            img = self._prepare_combined()
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.statusBar().showMessage(f"Guardado: {path}")

    # ── Descripción ───────────────────────────────────────────────────────────

    def _get_description(self) -> str:
        d = {
            "RGB":  "Modelo aditivo.\nRojo + Verde + Azul.",
            "CMYK": "Modelo sustractivo.\nCyan · Magenta · Yellow · Key.",
            "HSV":  "Modelo perceptual.\nMatiz · Saturación · Valor.",
            "HSI":  "Percepción humana.\nMatiz · Saturación · Intensidad.",
            "YUV":  "Video y televisión.\nLuminancia + Crominancia.",
            "LAB":  "CIE L*a*b*.\nSepara luminosidad del color.",
            "XYZ":  "CIE 1931.\nBase para conversiones.",
        }
        return d.get(self.model_name, "Sin descripción.")
