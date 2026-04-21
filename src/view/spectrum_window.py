"""SpectrumWindow — ventana dedicada para la Práctica 5 (FFT).

Muestra la grilla de 5 imágenes estándar de la práctica:
  [original] [magnitud] [fase]
  [vacío]    [máscara]  [filtrada]

Diseño coherente con HistogramWindow y ResultsWindow.

Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria
ESCOM · IPN — Análisis de Imágenes — Mayo 2026
"""
from __future__ import annotations
import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

# ── Paleta ────────────────────────────────────────────────────────────────────
BG      = "#1e2b3c"
BG_DARK = "#141e2b"
BG_CARD = "#243447"
BORDER  = "#2a3f57"
TEXT    = "#a8b8cc"
TEXT_HI = "#e8f0f8"
LABEL   = "#5a7a99"
ACCENT  = "#4a9eff"

# Colores de badge por tipo de imagen
_BADGE = {
    "espectro": ("#6bb5ff", "#001030"),
    "fase":     ("#cc88ff", "#1a0030"),
    "filtro":   ("#6bffa0", "#003018"),
    "resultado":("#ffaa44", "#2a1000"),
    "original": ("#a8b8cc", "#1e2b3c"),
}


def _np_to_pixmap(img: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qi = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        qi = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi).scaled(
        max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def _save_png(img: np.ndarray, path: str):
    if img.ndim == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, img)


# ── Tarjeta de imagen ─────────────────────────────────────────────────────────

class _ImgCard(QWidget):
    W, H = 280, 220

    def __init__(self, title: str, img: np.ndarray,
                 badge_key: str = "original", extra: str = ""):
        super().__init__()
        self._img = img
        bc, bb = _BADGE.get(badge_key, _BADGE["original"])
        self.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER};"
            f"border-radius: 8px;"
        )
        self.setFixedWidth(self.W + 16)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(5)

        # badge
        badge = QLabel(badge_key.upper())
        badge.setStyleSheet(
            f"color: {bc}; background: {bb}; border-radius: 3px;"
            f"font-size: 9px; font-weight: 700; padding: 1px 6px; border: none;"
        )
        brow = QHBoxLayout()
        brow.addWidget(badge)
        brow.addStretch()
        lay.addLayout(brow)

        # imagen
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(self.W, self.H)
        lbl.setStyleSheet(f"background: {BG_DARK}; border-radius: 4px; border: none;")
        if img is not None:
            lbl.setPixmap(_np_to_pixmap(img, self.W, self.H))
        lay.addWidget(lbl)

        # título
        t = QLabel(title)
        t.setAlignment(Qt.AlignCenter)
        t.setWordWrap(True)
        t.setStyleSheet(f"color: {TEXT_HI}; font-size: 11px; font-weight: 600;"
                        f"background: transparent; border: none;")
        lay.addWidget(t)

        # info extra
        if extra:
            e = QLabel(extra)
            e.setAlignment(Qt.AlignCenter)
            e.setStyleSheet(f"color: {LABEL}; font-size: 10px;"
                            f"background: transparent; border: none;")
            lay.addWidget(e)

        # botón guardar
        btn = QPushButton("💾  Guardar PNG")
        btn.setFixedHeight(26)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT};
                border: 1px solid {BORDER}; border-radius: 4px;
                font-size: 10px; padding: 0 8px;
            }}
            QPushButton:hover {{ background: {BG_CARD}; color: {TEXT_HI}; }}
        """)
        btn.clicked.connect(self._save)
        lay.addWidget(btn)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", "resultado.png", "PNG (*.png)"
        )
        if path and self._img is not None:
            _save_png(self._img, path)


# ── SpectrumWindow ─────────────────────────────────────────────────────────────

class SpectrumWindow(QMainWindow):
    """Ventana que muestra los 5 paneles estándar de análisis FFT:
    original · magnitud · fase · máscara · imagen filtrada.

    Args:
        original_rgb  : imagen original RGB uint8
        magnitud_arr  : espectro de magnitud float32 (log)
        fase_arr      : espectro de fase float32 [-π, π]
        mask_arr      : máscara float32 [0, 1]
        filtrada_rgb  : imagen reconstruida RGB uint8
        filtro        : nombre del filtro ('Butterworth', etc.)
        tipo          : 'Pasa bajas' o 'Pasa altas'
        cutoff        : valor de cutoff
        orden         : orden Butterworth
        psnr          : PSNR respecto al original (opcional)
    """

    def __init__(
        self,
        original_rgb: np.ndarray,
        magnitud_arr: np.ndarray,
        fase_arr: np.ndarray,
        mask_arr: np.ndarray,
        filtrada_rgb: np.ndarray,
        filtro: str = "Butterworth",
        tipo: str = "Pasa bajas",
        cutoff: float = 0.15,
        orden: int = 2,
        psnr: float | None = None,
    ):
        super().__init__()
        self.setWindowTitle(
            f"Práctica 5 — Espectro FFT  ·  {filtro} {tipo}  "
            f"cutoff={cutoff:.2f}  orden={orden}"
        )
        self.setMinimumSize(1050, 640)
        self.resize(1280, 720)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barra superior con parámetros ─────────────────────────────
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(f"background: {BG_CARD}; border-bottom: 1px solid {BORDER};")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 0, 16, 0)
        bl.setSpacing(24)
        for lbl_txt, val_txt in [
            ("Filtro",   filtro),
            ("Tipo",     tipo),
            ("Cutoff",   f"{cutoff:.2f}"),
            ("Orden",    str(orden)),
            ("PSNR",     f"{psnr:.2f} dB" if psnr is not None else "—"),
        ]:
            pair = QWidget()
            pair.setStyleSheet("background: transparent;")
            pl = QHBoxLayout(pair)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.setSpacing(4)
            l1 = QLabel(lbl_txt + ":")
            l1.setStyleSheet(f"color: {LABEL}; font-size: 11px; background: transparent;")
            l2 = QLabel(val_txt)
            l2.setStyleSheet(
                f"color: {TEXT_HI}; font-size: 11px; font-weight: 600;"
                f"background: transparent;"
            )
            pl.addWidget(l1)
            pl.addWidget(l2)
            bl.addWidget(pair)

        btn_export = QPushButton("📁  Exportar todo")
        btn_export.setFixedHeight(28)
        btn_export.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT}; color: #fff; border: none;
                border-radius: 5px; font-size: 11px; padding: 0 14px;
            }}
            QPushButton:hover {{ background: #3a8eef; }}
        """)
        self._imgs = {
            "original":  original_rgb,
            "magnitud":  self._float_to_uint8(magnitud_arr),
            "fase":      self._fase_to_rgb(fase_arr),
            "mascara":   self._float_to_uint8(mask_arr),
            "filtrada":  filtrada_rgb,
        }
        btn_export.clicked.connect(self._export_all)
        bl.addStretch()
        bl.addWidget(btn_export)
        root.addWidget(bar)

        # ── Grid de tarjetas ──────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG}; border: none; }}
            QScrollBar:vertical {{ background: {BG}; width: 6px; border: none; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
        """)
        content = QWidget()
        content.setStyleSheet(f"background: {BG};")
        grid = QGridLayout(content)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setSpacing(14)

        # Fila 0
        grid.addWidget(
            _ImgCard("Imagen original (escala de grises)", original_rgb,
                     "original"),
            0, 0
        )
        grid.addWidget(
            _ImgCard("Espectro de magnitud (log)", self._float_to_uint8(magnitud_arr),
                     "espectro", "log(1 + |F|)"),
            0, 1
        )
        grid.addWidget(
            _ImgCard("Espectro de fase", self._fase_to_rgb(fase_arr),
                     "fase", "ángulo en [−π, π]"),
            0, 2
        )
        # Fila 1 — posición (1,0) vacía, centrar visualmente con spacer
        spacer = QWidget()
        spacer.setStyleSheet(f"background: {BG};")
        grid.addWidget(spacer, 1, 0)
        grid.addWidget(
            _ImgCard(
                f"Máscara  {filtro} {tipo}",
                self._float_to_uint8(mask_arr),
                "filtro",
                f"cutoff={cutoff:.2f}  orden={orden}",
            ),
            1, 1
        )
        psnr_str = f"PSNR = {psnr:.2f} dB" if psnr is not None else ""
        grid.addWidget(
            _ImgCard("Imagen filtrada (IFFT)", filtrada_rgb,
                     "resultado", psnr_str),
            1, 2
        )

        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

        self.statusBar().setStyleSheet(
            f"background: {BG}; color: {LABEL}; font-size: 11px;"
        )
        self.statusBar().showMessage(
            f"Práctica 5 · FFT 2D · {filtro} {tipo} · cutoff={cutoff:.2f}"
        )

    # ── Helpers de conversión ─────────────────────────────────────────────────

    @staticmethod
    def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
        """float32 normalizado → RGB uint8."""
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            norm = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = np.zeros(arr.shape[:2], dtype=np.uint8)
        if norm.ndim == 2:
            return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        return norm

    @staticmethod
    def _fase_to_rgb(phase: np.ndarray) -> np.ndarray:
        """Fase [-π, π] → RGB con colormap TWILIGHT_SHIFTED."""
        import math as _m
        norm = ((phase + _m.pi) / (2 * _m.pi) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TWILIGHT_SHIFTED)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def _export_all(self):
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de destino")
        if not folder:
            return
        import os
        saved = 0
        for name, img in self._imgs.items():
            if img is not None:
                _save_png(img, os.path.join(folder, f"fft_{name}.png"))
                saved += 1
        self.statusBar().showMessage(f"✓ {saved} imágenes exportadas en: {folder}")
