"""ResultsWindow — ventana dedicada para mostrar resultados de las prácticas 3-a/b/c.

Diseño coherente con HistogramWindow:
  · Fondo azul oscuro
  · Panel izquierdo: imagen de referencia + info/estadísticas
  · Panel derecho: grid de resultados con imagen + título + badge
  · Botón "Guardar PNG" por cada resultado individual
  · Botón "Exportar todo" que guarda todas las imágenes en una carpeta

Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria
ESCOM · IPN — Análisis de Imágenes — Marzo 2026
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

# ── Paleta (misma que el resto de la app) ────────────────────────────────────
BG       = "#1e2b3c"
BG_DARK  = "#141e2b"
BG_CARD  = "#243447"
BORDER   = "#2a3f57"
TEXT     = "#a8b8cc"
TEXT_HI  = "#e8f0f8"
LABEL    = "#5a7a99"
ACCENT   = "#4a9eff"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_pixmap(img: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """Convierte array RGB/gris numpy → QPixmap escalado."""
    if img is None:
        return QPixmap()
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qi = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        qi = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi).scaled(
        max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def _save_png(img: np.ndarray, filepath: str):
    """Guarda array RGB/gris como PNG sin pérdida."""
    if img.ndim == 3:
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(filepath, img)


def _accent_btn(text: str, color: str = ACCENT) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(28)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {color}; color: #fff;
            border: none; border-radius: 5px;
            font-size: 11px; padding: 0 12px;
        }}
        QPushButton:hover {{ background: #3a8eef; }}
        QPushButton:pressed {{ background: #2a7edf; }}
    """)
    return btn


def _ghost_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(26)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: transparent; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 4px;
            font-size: 10px; padding: 0 8px;
        }}
        QPushButton:hover {{ background: {BG_CARD}; color: {TEXT_HI}; }}
    """)
    return btn


# ── Tarjeta de resultado individual ──────────────────────────────────────────

class _ResultCard(QWidget):
    """Widget que muestra una imagen resultado con título, badge y botón guardar."""

    THUMB_W = 280
    THUMB_H = 220

    # Colores de badge según categoría
    _BADGE_COLORS = {
        "ruido":      ("#ff9944", "#3a1800"),
        "lógica":     ("#6bb5ff", "#001830"),
        "aritmética": ("#6bffa0", "#003018"),
        "relacional": ("#cc88ff", "#1a0030"),
        "etiquetado": ("#ffee44", "#2a2000"),
        "default":    ("#4a9eff", "#001428"),
    }

    def __init__(self, title: str, img: np.ndarray,
                 badge: str = "default", extra_info: str = ""):
        super().__init__()
        self._img = img
        self._title = title
        self.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 8px;"
        )
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedWidth(self.THUMB_W + 16)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Badge ─────────────────────────────────────────────────────
        badge_color, badge_bg = self._BADGE_COLORS.get(
            badge, self._BADGE_COLORS["default"]
        )
        badge_row = QHBoxLayout()
        badge_row.setSpacing(4)
        badge_lbl = QLabel(badge.upper())
        badge_lbl.setStyleSheet(
            f"color: {badge_color}; background: {badge_bg};"
            f"border-radius: 3px; font-size: 9px; font-weight: 700;"
            f"padding: 1px 6px; border: none;"
        )
        badge_row.addWidget(badge_lbl)
        badge_row.addStretch()
        layout.addLayout(badge_row)

        # ── Imagen ────────────────────────────────────────────────────
        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setFixedSize(self.THUMB_W, self.THUMB_H)
        img_lbl.setStyleSheet(f"background: {BG_DARK}; border-radius: 4px; border: none;")
        if img is not None:
            img_lbl.setPixmap(_to_pixmap(img, self.THUMB_W, self.THUMB_H))
        layout.addWidget(img_lbl)

        # ── Título ────────────────────────────────────────────────────
        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setWordWrap(True)
        title_lbl.setStyleSheet(
            f"color: {TEXT_HI}; font-size: 11px; font-weight: 600;"
            f"background: transparent; border: none;"
        )
        layout.addWidget(title_lbl)

        # ── Info extra ────────────────────────────────────────────────
        if extra_info:
            info_lbl = QLabel(extra_info)
            info_lbl.setAlignment(Qt.AlignCenter)
            info_lbl.setWordWrap(True)
            info_lbl.setStyleSheet(
                f"color: {LABEL}; font-size: 10px; background: transparent; border: none;"
            )
            layout.addWidget(info_lbl)

        # ── Botón guardar ─────────────────────────────────────────────
        btn_save = _ghost_btn("💾  Guardar PNG")
        btn_save.clicked.connect(self._save_single)
        layout.addWidget(btn_save)

    def _save_single(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar resultado",
            f"{self._title.replace(' ', '_')}.png",
            "PNG sin pérdida (*.png)",
        )
        if path:
            _save_png(self._img, path)


# ── Panel izquierdo ───────────────────────────────────────────────────────────

class _LeftPanel(QWidget):
    """Panel con imagen de referencia + lista de estadísticas."""

    def __init__(self, ref_img: np.ndarray, title: str, info_rows: list[tuple]):
        super().__init__()
        self.setFixedWidth(230)
        self.setStyleSheet(f"background: {BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Encabezado
        hdr = QLabel(f"  {title}")
        hdr.setFixedHeight(40)
        hdr.setStyleSheet(
            f"background: {BG_CARD}; color: {TEXT_HI}; font-size: 13px;"
            f"font-weight: 600; border-bottom: 2px solid {ACCENT};"
        )
        layout.addWidget(hdr)

        # Miniatura
        if ref_img is not None:
            thumb = QLabel()
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setFixedHeight(200)
            thumb.setStyleSheet(f"background: {BG_DARK};")
            thumb.setPixmap(_to_pixmap(ref_img, 220, 196))
            layout.addWidget(thumb)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background: {BORDER}; max-height: 1px;")
        layout.addWidget(sep)

        # Filas de info
        for lbl_txt, val_txt in info_rows:
            row = QWidget()
            row.setStyleSheet("background: transparent;")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(12, 4, 12, 4)
            l1 = QLabel(lbl_txt)
            l1.setStyleSheet(f"color: {LABEL}; font-size: 10px; background: transparent;")
            l2 = QLabel(val_txt)
            l2.setStyleSheet(
                f"color: {TEXT_HI}; font-size: 10px; font-weight: 600; background: transparent;"
            )
            l2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            l2.setWordWrap(True)
            rl.addWidget(l1)
            rl.addWidget(l2)
            layout.addWidget(row)

        layout.addStretch()


# ══════════════════════════════════════════════════════════════════════════════
# ResultsWindow
# ══════════════════════════════════════════════════════════════════════════════

class ResultsWindow(QMainWindow):
    """Ventana genérica para mostrar un grid de resultados de imagen.

    Args:
        title      : título de la ventana.
        ref_img    : imagen de referencia (mostrada en panel izquierdo).
        cards_data : lista de dicts con claves:
                       "title"      (str)  — nombre de la tarjeta
                       "img"        (ndarray) — imagen resultado RGB/gris
                       "badge"      (str)  — categoría ("ruido","lógica",…)
                       "extra_info" (str, opcional) — texto extra bajo el título
        left_title : título del panel izquierdo (defecto "Imagen original")
        left_info  : lista de (label, value) para el panel izquierdo
        cols       : columnas del grid (defecto 3)
    """

    def __init__(
        self,
        title: str,
        ref_img: np.ndarray,
        cards_data: List[dict],
        left_title: str = "Imagen original",
        left_info: list[tuple] | None = None,
        cols: int = 3,
    ):
        super().__init__()
        self._cards_data = cards_data
        self.setWindowTitle(title)
        self.setMinimumSize(980, 620)
        self.resize(1200, 720)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Panel izquierdo
        root.addWidget(
            _LeftPanel(ref_img, left_title, left_info or [])
        )

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {BORDER};")
        root.addWidget(sep)

        # Panel derecho
        root.addWidget(self._build_right(cards_data, cols), stretch=1)

        # Status bar
        self.statusBar().setStyleSheet(
            f"background: {BG}; color: {LABEL}; font-size: 11px;"
        )
        self.statusBar().showMessage(
            f"{title}  ·  {len(cards_data)} resultado(s)"
        )

    # ── Panel derecho ─────────────────────────────────────────────────────────

    def _build_right(self, cards_data: list, cols: int) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {BG};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar superior con botón "Exportar todo"
        bar = QWidget()
        bar.setFixedHeight(44)
        bar.setStyleSheet(f"background: {BG_CARD}; border-bottom: 1px solid {BORDER};")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(16, 0, 16, 0)
        bar_layout.setSpacing(8)

        count_lbl = QLabel(f"{len(cards_data)} resultado(s)")
        count_lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; background: transparent;")
        bar_layout.addWidget(count_lbl)
        bar_layout.addStretch()

        btn_export_all = _accent_btn("📁  Exportar todo como PNG")
        btn_export_all.clicked.connect(self._export_all)
        bar_layout.addWidget(btn_export_all)
        layout.addWidget(bar)

        # Área scrollable con grid de tarjetas
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG}; border: none; }}
            QScrollBar:vertical {{
                background: {BG}; width: 6px; border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER}; border-radius: 3px; min-height: 20px;
            }}
            QScrollBar:horizontal {{
                background: {BG}; height: 6px; border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {BORDER}; border-radius: 3px;
            }}
        """)

        content = QWidget()
        content.setStyleSheet(f"background: {BG};")
        grid = QGridLayout(content)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setSpacing(14)

        for i, data in enumerate(cards_data):
            card = _ResultCard(
                title=data.get("title", f"Resultado {i+1}"),
                img=data.get("img"),
                badge=data.get("badge", "default"),
                extra_info=data.get("extra_info", ""),
            )
            grid.addWidget(card, i // cols, i % cols)

        # Alinear última fila a la izquierda si está incompleta
        for c in range(cols):
            grid.setColumnStretch(c, 0)
        grid.setColumnStretch(cols, 1)   # columna fantasma que absorbe espacio

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)
        return panel

    # ── Exportar todo ─────────────────────────────────────────────────────────

    def _export_all(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de destino"
        )
        if not folder:
            return
        saved = 0
        for data in self._cards_data:
            img = data.get("img")
            if img is None:
                continue
            name = data.get("title", f"resultado_{saved}").replace(" ", "_")
            name = "".join(c for c in name if c.isalnum() or c in "_-")
            path = os.path.join(folder, f"{name}.png")
            _save_png(img, path)
            saved += 1
        self.statusBar().showMessage(
            f"✓  {saved} imágenes exportadas en: {folder}"
        )
