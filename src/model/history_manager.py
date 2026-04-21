"""HistoryManager — historial de imágenes procesadas durante la sesión.

Cada vez que el controlador llama a _store_result(), registra una entrada
con la imagen resultante, el nombre de la operación y la práctica a la que
pertenece. La ventana HistoryWindow consulta este manager para mostrar
el historial completo.

No depende de Qt: es un objeto de modelo puro.

Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria
ESCOM · IPN — Análisis de Imágenes — Mayo 2026
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


# Mapa de prefijos de clave → nombre de práctica legible
_PRACTICA_MAP: dict[str, str] = {
    "BINARIA":    "P1 · Binarización",
    "MAPA":       "P1 · Mapa color",
    "Personaliz": "P1 · Mapa color",
    "RUIDO":      "P3a · Ruido",
    "FILTRO":     "P3a · Filtro",
    "ARIT":       "P3b · Aritmética",
    "LOGICA":     "P3b · Lógica",
    "REL":        "P3b · Relacional",
    "CC":         "P3c · Comp. conexas",
    "MORPH":      "P4 · Morfología",
    "FFT":        "P5 · FFT",
    "DCT":        "P5 · DCT",
}

# Colores badge por práctica (hex sin #)
_PRACTICA_COLOR: dict[str, str] = {
    "P1":  "4a9eff",
    "P3a": "ff9944",
    "P3b": "6bffa0",
    "P3c": "ffee44",
    "P4":  "cc88ff",
    "P5":  "6bb5ff",
}


def _practica_from_key(key: str) -> str:
    for prefix, name in _PRACTICA_MAP.items():
        if key.upper().startswith(prefix.upper()):
            return name
    return "General"


def _color_from_practica(practica: str) -> str:
    for code, color in _PRACTICA_COLOR.items():
        if practica.startswith(code):
            return color
    return "4a9eff"


@dataclass
class HistoryEntry:
    """Una entrada del historial: imagen + metadata."""
    index:     int
    timestamp: datetime
    image_name: str          # nombre de la imagen base (current_image)
    op_key:    str           # clave interna, e.g. "MORPH_EROSION"
    practica:  str           # nombre legible, e.g. "P4 · Morfología"
    color:     str           # hex sin # para el badge
    img_rgb:   np.ndarray    # array RGB uint8

    @property
    def label(self) -> str:
        """Etiqueta corta para la lista."""
        op = self.op_key.replace("_", " ").title()
        return op

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    @property
    def full_label(self) -> str:
        return f"{self.time_str}  ·  {self.label}"


class HistoryManager:
    """Almacena todas las imágenes procesadas durante la sesión.

    Uso:
        manager = HistoryManager()
        manager.add(image_name, op_key, img_rgb)
        entries = manager.all()        # lista completa
        entry   = manager.get(idx)     # por índice
        manager.clear()                # limpiar sesión
    """

    def __init__(self):
        self._entries: list[HistoryEntry] = []
        self._counter: int = 0

    def add(self, image_name: str, op_key: str,
            img_rgb: np.ndarray) -> HistoryEntry:
        """Registra una nueva entrada. Devuelve la entrada creada."""
        practica = _practica_from_key(op_key)
        color    = _color_from_practica(practica)
        entry = HistoryEntry(
            index=self._counter,
            timestamp=datetime.now(),
            image_name=image_name,
            op_key=op_key,
            practica=practica,
            color=color,
            img_rgb=img_rgb.copy() if img_rgb is not None else np.zeros((4, 4, 3), np.uint8),
        )
        self._entries.append(entry)
        self._counter += 1
        return entry

    def all(self) -> list[HistoryEntry]:
        """Devuelve todas las entradas, más reciente primero."""
        return list(reversed(self._entries))

    def get(self, index: int) -> HistoryEntry | None:
        """Devuelve entrada por índice absoluto."""
        for e in self._entries:
            if e.index == index:
                return e
        return None

    def by_practica(self, practica_prefix: str) -> list[HistoryEntry]:
        """Filtra entradas que empiecen con el prefijo de práctica dado."""
        return [e for e in self.all() if e.practica.startswith(practica_prefix)]

    def clear(self):
        self._entries.clear()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._entries)
