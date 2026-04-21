"""HistoryWindow — ventana de historial de imágenes procesadas.

Diseño:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  TOOLBAR: filtro de práctica · búsqueda · botones de acción            │
  ├──────────────────────┬─────────────────────────────────────────────────┤
  │  PANEL IZQUIERDO     │  PANEL DERECHO                                  │
  │  Lista de entradas   │  ┌──────────────────────────────────────────┐   │
  │  con badge práctica  │  │  Visor grande de la imagen seleccionada  │   │
  │  y timestamp         │  └──────────────────────────────────────────┘   │
  │                      │  Metadata: operación, práctica, tiempo, tamaño  │
  │                      │  Botones: Ver original · Ver comparar           │
  │                      │           Usar como base · Guardar PNG           │
  └──────────────────────┴─────────────────────────────────────────────────┘
  Pestaña 2: Grid de todas las imágenes (thumbnails 160×120 con label)

Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria
ESCOM · IPN — Análisis de Imágenes — Mayo 2026
"""
from __future__ import annotations
import os
from typing import Callable

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QTabWidget,
    QVBoxLayout, QWidget, QComboBox,
)

from model.history_manager import HistoryEntry, HistoryManager

# ── Paleta ────────────────────────────────────────────────────────────────────
BG       = "#1e2b3c"
BG_DARK  = "#141e2b"
BG_CARD  = "#243447"
BORDER   = "#2a3f57"
TEXT     = "#a8b8cc"
TEXT_HI  = "#e8f0f8"
LABEL    = "#5a7a99"
ACCENT   = "#4a9eff"
HOVER    = "#263548"
SELECTED = "#2d4a6b"

_FILTER_OPTIONS = [
    "Todas",
    "P1 · Binarización / Mapa",
    "P3a · Ruido / Filtro",
    "P3b · Aritmética / Lógica / Relacional",
    "P3c · Comp. conexas",
    "P4 · Morfología",
    "P5 · FFT / DCT",
]

_FILTER_PREFIXES = {
    "Todas": "",
    "P1 · Binarización / Mapa": "P1",
    "P3a · Ruido / Filtro":     "P3a",
    "P3b · Aritmética / Lógica / Relacional": "P3b",
    "P3c · Comp. conexas":      "P3c",
    "P4 · Morfología":          "P4",
    "P5 · FFT / DCT":           "P5",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_pixmap(img: np.ndarray, w: int, h: int) -> QPixmap:
    img = np.ascontiguousarray(img)
    hi, wi = img.shape[:2]
    if img.ndim == 2:
        qi = QImage(img.data, wi, hi, wi, QImage.Format_Grayscale8)
    else:
        qi = QImage(img.data, wi, hi, 3 * wi, QImage.Format_RGB888)
    return QPixmap.fromImage(qi).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _accent_btn(text: str, color: str = ACCENT) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(30)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {color}; color: #fff;
            border: none; border-radius: 5px;
            font-size: 11px; padding: 0 14px;
        }}
        QPushButton:hover {{ background: #3a8eef; }}
        QPushButton:pressed {{ background: #2a7edf; }}
        QPushButton:disabled {{ background: {BORDER}; color: {LABEL}; }}
    """)
    return btn


def _ghost_btn(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(28)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: transparent; color: {TEXT};
            border: 1px solid {BORDER}; border-radius: 4px;
            font-size: 10px; padding: 0 10px;
        }}
        QPushButton:hover {{ background: {BG_CARD}; color: {TEXT_HI}; }}
        QPushButton:disabled {{ color: {LABEL}; border-color: {BG_CARD}; }}
    """)
    return btn


# ── Thumbnail del grid ────────────────────────────────────────────────────────

class _Thumb(QWidget):
    """Miniatura clicable para el grid de todas las imágenes."""

    clicked = pyqtSignal(int)   # emite el índice de la entrada

    TW, TH = 164, 126

    def __init__(self, entry: HistoryEntry):
        super().__init__()
        self._idx = entry.index
        self.setFixedSize(self.TW + 10, self.TH + 52)
        self.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER};"
            f"border-radius: 7px;"
        )
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(4)

        # Badge práctica
        badge = QLabel(entry.practica)
        badge.setStyleSheet(
            f"color: #{entry.color}; background: transparent;"
            f"font-size: 9px; font-weight: 700; border: none; padding: 0;"
        )
        lay.addWidget(badge)

        # Imagen
        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignCenter)
        img_lbl.setFixedSize(self.TW, self.TH)
        img_lbl.setStyleSheet(f"background: {BG_DARK}; border-radius: 4px; border: none;")
        img_lbl.setPixmap(_to_pixmap(entry.img_rgb, self.TW, self.TH))
        lay.addWidget(img_lbl)

        # Label operación (truncada)
        op = entry.label
        if len(op) > 22:
            op = op[:20] + "…"
        lbl = QLabel(op)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"color: {TEXT_HI}; font-size: 10px; font-weight: 600;"
            f"background: transparent; border: none;"
        )
        lay.addWidget(lbl)

        # Timestamp
        ts = QLabel(entry.time_str)
        ts.setAlignment(Qt.AlignCenter)
        ts.setStyleSheet(f"color: {LABEL}; font-size: 9px; background: transparent; border: none;")
        lay.addWidget(ts)

    def mousePressEvent(self, _):
        self.clicked.emit(self._idx)

    def enterEvent(self, _):
        self.setStyleSheet(
            f"background: {HOVER}; border: 1px solid {ACCENT};"
            f"border-radius: 7px;"
        )

    def leaveEvent(self, _):
        self.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER};"
            f"border-radius: 7px;"
        )


# ── Panel derecho: detalle de una entrada ────────────────────────────────────

class _DetailPanel(QWidget):
    """Muestra el detalle de la entrada seleccionada en la lista."""

    # Señales emitidas hacia el controlador a través de HistoryWindow
    sig_show_original  = pyqtSignal(np.ndarray)
    sig_show_compare   = pyqtSignal(np.ndarray)
    sig_use_as_base    = pyqtSignal(np.ndarray, str)   # img, op_key

    IMG_W, IMG_H = 560, 440

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {BG};")
        self._entry: HistoryEntry | None = None

        main = QVBoxLayout(self)
        main.setContentsMargins(16, 12, 16, 12)
        main.setSpacing(10)

        # Visor grande
        self._viewer = QLabel()
        self._viewer.setAlignment(Qt.AlignCenter)
        self._viewer.setMinimumSize(400, 320)
        self._viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._viewer.setStyleSheet(
            f"background: {BG_DARK}; border-radius: 8px; border: 1px solid {BORDER};"
        )
        self._viewer.setText(
            f'<span style="color:{LABEL}; font-size:13px;">'
            f'Selecciona una entrada del historial</span>'
        )
        main.addWidget(self._viewer, stretch=1)

        # Metadata row
        self._meta_widget = QWidget()
        self._meta_widget.setStyleSheet(f"background: {BG_CARD}; border-radius: 6px;")
        meta_lay = QHBoxLayout(self._meta_widget)
        meta_lay.setContentsMargins(12, 8, 12, 8)
        meta_lay.setSpacing(24)
        self._pairs: dict[str, QLabel] = {}
        for key in ("Operación", "Práctica", "Imagen base", "Hora", "Tamaño"):
            col = QWidget()
            col.setStyleSheet("background: transparent;")
            cl = QVBoxLayout(col)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            lk = QLabel(key)
            lk.setStyleSheet(f"color: {LABEL}; font-size: 9px; background: transparent;")
            lv = QLabel("—")
            lv.setStyleSheet(f"color: {TEXT_HI}; font-size: 11px; font-weight: 600; background: transparent;")
            cl.addWidget(lk)
            cl.addWidget(lv)
            meta_lay.addWidget(col)
            self._pairs[key] = lv
        meta_lay.addStretch()
        main.addWidget(self._meta_widget)

        # Botones de acción
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._btn_orig  = _accent_btn("📌  Ver en Original")
        self._btn_cmp   = _accent_btn("⚖  Ver en Comparar", "#2d6a4f")
        self._btn_base  = _ghost_btn("🔁  Usar como base")
        self._btn_save  = _ghost_btn("💾  Guardar PNG")

        for b in (self._btn_orig, self._btn_cmp, self._btn_base, self._btn_save):
            b.setEnabled(False)
            btn_row.addWidget(b)
        btn_row.addStretch()
        main.addLayout(btn_row)

        self._btn_orig.clicked.connect(self._on_show_original)
        self._btn_cmp.clicked.connect(self._on_show_compare)
        self._btn_base.clicked.connect(self._on_use_as_base)
        self._btn_save.clicked.connect(self._on_save)

    def load(self, entry: HistoryEntry):
        self._entry = entry
        h, w = entry.img_rgb.shape[:2]

        # Imagen
        px = _to_pixmap(entry.img_rgb, self.IMG_W, self.IMG_H)
        self._viewer.setPixmap(px)

        # Metadata
        self._pairs["Operación"].setText(entry.label)
        self._pairs["Práctica"].setText(entry.practica)
        self._pairs["Imagen base"].setText(entry.image_name)
        self._pairs["Hora"].setText(entry.time_str)
        self._pairs["Tamaño"].setText(f"{w} × {h} px")

        for b in (self._btn_orig, self._btn_cmp, self._btn_base, self._btn_save):
            b.setEnabled(True)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._entry is not None:
            aw = max(200, self._viewer.width() - 4)
            ah = max(200, self._viewer.height() - 4)
            self._viewer.setPixmap(_to_pixmap(self._entry.img_rgb, aw, ah))

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_show_original(self):
        if self._entry:
            self.sig_show_original.emit(self._entry.img_rgb)

    def _on_show_compare(self):
        if self._entry:
            self.sig_show_compare.emit(self._entry.img_rgb)

    def _on_use_as_base(self):
        if self._entry:
            self.sig_use_as_base.emit(self._entry.img_rgb, self._entry.op_key)

    def _on_save(self):
        if not self._entry:
            return
        default = self._entry.op_key.lower().replace(" ", "_") + ".png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar imagen", default, "PNG sin pérdida (*.png)"
        )
        if path:
            img = self._entry.img_rgb
            if img.ndim == 3:
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(path, img)


# ── HistoryWindow ─────────────────────────────────────────────────────────────

class HistoryWindow(QMainWindow):
    """Ventana principal del historial de imágenes procesadas.

    Conecta con el controlador a través de tres callbacks opcionales:
        on_show_original(img_rgb)      → mostrar en pestaña Original
        on_show_compare(img_rgb)       → mostrar en pestaña Comparar
        on_use_as_base(img_rgb, key)   → usar como imagen activa
    """

    def __init__(
        self,
        history: HistoryManager,
        on_show_original: Callable | None = None,
        on_show_compare:  Callable | None = None,
        on_use_as_base:   Callable | None = None,
    ):
        super().__init__()
        self._history = history
        self._cb_orig = on_show_original
        self._cb_cmp  = on_show_compare
        self._cb_base = on_use_as_base

        self.setWindowTitle("Historial de imágenes procesadas")
        self.setMinimumSize(1100, 660)
        self.resize(1340, 760)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Barra superior ─────────────────────────────────────────────────
        root.addWidget(self._build_toolbar())

        # ── Tabs: Lista+Detalle  /  Grid ───────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {BG}; }}
            QTabBar::tab {{
                background: {BG}; color: {LABEL};
                padding: 8px 22px; font-size: 12px; border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {TEXT_HI}; background: {BG_CARD};
                border-bottom: 2px solid {ACCENT};
            }}
            QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
        """)

        self._detail = _DetailPanel()
        self._detail.sig_show_original.connect(self._on_show_original)
        self._detail.sig_show_compare.connect(self._on_show_compare)
        self._detail.sig_use_as_base.connect(self._on_use_as_base)

        self._tabs.addTab(self._build_list_tab(), "  Lista + Detalle  ")
        self._tabs.addTab(self._build_grid_tab(), "  Grid de todas  ")
        root.addWidget(self._tabs, stretch=1)

        # Status bar
        self.statusBar().setStyleSheet(
            f"background: {BG}; color: {LABEL}; font-size: 11px;"
        )
        self._update_status()

    # ── Barra de toolbar ──────────────────────────────────────────────────────

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet(
            f"background: {BG_CARD}; border-bottom: 1px solid {BORDER};"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(12)

        # Filtro por práctica
        lbl = QLabel("Práctica:")
        lbl.setStyleSheet(f"color: {LABEL}; font-size: 11px; background: transparent;")
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(_FILTER_OPTIONS)
        self._filter_combo.setFixedWidth(240)
        self._filter_combo.setStyleSheet(f"""
            QComboBox {{
                background: {SELECTED}; color: {TEXT_HI};
                border: 1px solid {BORDER}; border-radius: 4px;
                padding: 4px 8px; font-size: 11px;
            }}
            QComboBox::drop-down {{ border: none; width: 18px; }}
            QComboBox QAbstractItemView {{
                background: {BG}; color: {TEXT};
                selection-background-color: {SELECTED};
            }}
        """)
        self._filter_combo.currentTextChanged.connect(self._refresh_list)

        # Búsqueda
        self._search = QLineEdit()
        self._search.setPlaceholderText("Buscar operación…")
        self._search.setFixedWidth(200)
        self._search.setStyleSheet(f"""
            QLineEdit {{
                background: {SELECTED}; color: {TEXT_HI};
                border: 1px solid {BORDER}; border-radius: 4px;
                padding: 4px 8px; font-size: 11px;
            }}
        """)
        self._search.textChanged.connect(self._refresh_list)

        # Contador
        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet(
            f"color: {LABEL}; font-size: 11px; background: transparent;"
        )

        # Botones
        btn_refresh   = _ghost_btn("↺  Actualizar")
        btn_clear     = _ghost_btn("🗑  Limpiar historial")
        btn_export_all = _accent_btn("📁  Exportar todo")

        btn_refresh.clicked.connect(self.refresh)
        btn_clear.clicked.connect(self._on_clear)
        btn_export_all.clicked.connect(self._on_export_all)

        for w in (lbl, self._filter_combo, self._search,
                  self._count_lbl):
            lay.addWidget(w)
        lay.addStretch()
        for w in (btn_refresh, btn_clear, btn_export_all):
            lay.addWidget(w)

        return bar

    # ── Pestaña 1: Lista + Detalle ────────────────────────────────────────────

    def _build_list_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(f"background: {BG};")

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        # Panel izquierdo: lista
        left = QWidget()
        left.setMinimumWidth(280)
        left.setMaximumWidth(360)
        left.setStyleSheet(f"background: {BG};")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        self._list = QListWidget()
        self._list.setStyleSheet(f"""
            QListWidget {{
                background: {BG}; border: none; outline: none;
                font-size: 11px;
            }}
            QListWidget::item {{
                background: {BG}; color: {TEXT};
                padding: 0px; border: none;
                border-bottom: 1px solid {BORDER};
            }}
            QListWidget::item:selected {{
                background: {SELECTED}; color: {TEXT_HI};
            }}
            QListWidget::item:hover:!selected {{
                background: {HOVER};
            }}
        """)
        self._list.currentRowChanged.connect(self._on_list_select)
        ll.addWidget(self._list)
        splitter.addWidget(left)

        # Panel derecho: detalle
        splitter.addWidget(self._detail)
        splitter.setSizes([300, 900])

        lay = QHBoxLayout(widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(splitter)
        return widget

    def _make_list_item(self, entry: HistoryEntry) -> QListWidgetItem:
        """Crea un QListWidgetItem con widget personalizado para la entrada."""
        item = QListWidgetItem()
        item.setData(Qt.UserRole, entry.index)
        item.setSizeHint(QSize(260, 72))

        # Widget visual del ítem
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        hl = QHBoxLayout(w)
        hl.setContentsMargins(10, 6, 10, 6)
        hl.setSpacing(10)

        # Thumbnail pequeño
        thumb = QLabel()
        thumb.setFixedSize(54, 42)
        thumb.setStyleSheet(f"background: {BG_DARK}; border-radius: 3px;")
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setPixmap(_to_pixmap(entry.img_rgb, 54, 42))
        hl.addWidget(thumb)

        # Texto
        txt_col = QWidget()
        txt_col.setStyleSheet("background: transparent;")
        tl = QVBoxLayout(txt_col)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)

        badge_op = QHBoxLayout()
        badge_op.setSpacing(6)
        badge = QLabel(entry.practica)
        badge.setStyleSheet(
            f"color: #{entry.color}; background: transparent;"
            f"font-size: 9px; font-weight: 700;"
        )
        badge_op.addWidget(badge)
        badge_op.addStretch()

        op_lbl = QLabel(entry.label)
        op_lbl.setStyleSheet(
            f"color: {TEXT_HI}; font-size: 11px; font-weight: 600;"
            f"background: transparent;"
        )
        ts_lbl = QLabel(f"#{entry.index + 1}  ·  {entry.time_str}  ·  {entry.image_name}")
        ts_lbl.setStyleSheet(f"color: {LABEL}; font-size: 10px; background: transparent;")

        tl.addLayout(badge_op)
        tl.addWidget(op_lbl)
        tl.addWidget(ts_lbl)
        hl.addWidget(txt_col, stretch=1)

        return item, w

    # ── Pestaña 2: Grid ───────────────────────────────────────────────────────

    def _build_grid_tab(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet(f"background: {BG};")
        self._grid_outer = QVBoxLayout(widget)
        self._grid_outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG}; border: none; }}
            QScrollBar:vertical {{ background: {BG}; width: 6px; border: none; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
        """)
        self._grid_content = QWidget()
        self._grid_content.setStyleSheet(f"background: {BG};")
        self._grid_layout = QGridLayout(self._grid_content)
        self._grid_layout.setContentsMargins(16, 16, 16, 16)
        self._grid_layout.setSpacing(12)

        scroll.setWidget(self._grid_content)
        self._grid_outer.addWidget(scroll)
        return widget

    # ── Refresco de contenido ─────────────────────────────────────────────────

    def _filtered_entries(self) -> list[HistoryEntry]:
        prefix  = _FILTER_PREFIXES.get(self._filter_combo.currentText(), "")
        search  = self._search.text().strip().lower()
        entries = self._history.all()
        if prefix:
            entries = [e for e in entries if e.practica.startswith(prefix)]
        if search:
            entries = [e for e in entries
                       if search in e.label.lower()
                       or search in e.practica.lower()
                       or search in e.image_name.lower()]
        return entries

    def _refresh_list(self):
        entries = self._filtered_entries()
        self._list.clear()
        for entry in entries:
            item, widget = self._make_list_item(entry)
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)
        self._update_status(len(entries))

    def _refresh_grid(self):
        # Limpiar grid anterior
        while self._grid_layout.count():
            child = self._grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        entries = self._filtered_entries()
        cols = 5
        for i, entry in enumerate(entries):
            thumb = _Thumb(entry)
            thumb.clicked.connect(self._on_grid_click)
            self._grid_layout.addWidget(thumb, i // cols, i % cols)

        # Columna fantasma para alinear a la izquierda
        self._grid_layout.setColumnStretch(cols, 1)

    def refresh(self):
        """Actualiza lista y grid con el estado actual del historial."""
        self._refresh_list()
        self._refresh_grid()
        self._update_status()

    def _update_status(self, shown: int | None = None):
        total = len(self._history)
        if shown is None:
            shown = total
        self.statusBar().showMessage(
            f"Total en historial: {total}  ·  Mostrando: {shown}"
        )
        self._count_lbl.setText(f"{shown} / {total} entradas")

    # ── Eventos de selección ──────────────────────────────────────────────────

    def _on_list_select(self, row: int):
        item = self._list.item(row)
        if item is None:
            return
        idx = item.data(Qt.UserRole)
        entry = self._history.get(idx)
        if entry:
            self._detail.load(entry)

    def _on_grid_click(self, idx: int):
        entry = self._history.get(idx)
        if entry:
            self._detail.load(entry)
            # Cambiar a pestaña de detalle y seleccionar en lista
            self._tabs.setCurrentIndex(0)
            for row in range(self._list.count()):
                item = self._list.item(row)
                if item and item.data(Qt.UserRole) == idx:
                    self._list.setCurrentRow(row)
                    break

    # ── Callbacks del controlador ─────────────────────────────────────────────

    def _on_show_original(self, img: np.ndarray):
        if self._cb_orig:
            self._cb_orig(img)

    def _on_show_compare(self, img: np.ndarray):
        if self._cb_cmp:
            self._cb_cmp(img)

    def _on_use_as_base(self, img: np.ndarray, key: str):
        if self._cb_base:
            self._cb_base(img, key)

    # ── Acciones globales ─────────────────────────────────────────────────────

    def _on_clear(self):
        from PyQt5.QtWidgets import QMessageBox
        resp = QMessageBox.question(
            self, "Limpiar historial",
            "¿Eliminar todas las entradas del historial de esta sesión?\n"
            "Las imágenes procesadas en el modelo no se borran.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if resp == QMessageBox.Yes:
            self._history.clear()
            self.refresh()

    def _on_export_all(self):
        entries = self._filtered_entries()
        if not entries:
            return
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de destino")
        if not folder:
            return
        saved = 0
        for entry in entries:
            fname = f"{entry.index:03d}_{entry.op_key.lower()}.png"
            path  = os.path.join(folder, fname)
            img   = entry.img_rgb
            if img.ndim == 3:
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(path, img)
            saved += 1
        self.statusBar().showMessage(
            f"✓  {saved} imágenes exportadas en: {folder}"
        )
