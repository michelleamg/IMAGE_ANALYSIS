"""Ventana para mostrar modelos de color.
- Versión: 2.1 (bugfix)
  ✓ cmyk_to_rgb_display: clamp correcto de K para evitar valores
    negativos cuando k > 1 - epsilon tras la normalización
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QTabWidget, QPushButton, QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np


class ColorModelWindow(QMainWindow):
    def __init__(self, model_data, model_name, original_image):
        super().__init__()
        self.model_data = model_data
        self.model_name = model_name
        self.original_image = original_image

        self.setWindowTitle(f"Modelo de Color: {model_name}")
        self.setGeometry(150, 150, 1300, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        layout.addWidget(self.create_info_panel(), 1)
        layout.addWidget(self.create_channels_panel(), 3)

        self.statusBar().showMessage(f"Modelo {self.model_name} cargado correctamente")

    # ------------------------------------------------------------------
    # Paneles
    # ------------------------------------------------------------------

    def create_info_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)

        title = QLabel(f"Modelo: {self.model_name}")
        title.setStyleSheet("""
            font-size: 18px; font-weight: bold; padding: 10px;
            background-color: #4a90e2; color: white; border-radius: 5px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        channels = [k for k in self.model_data if k != "combined"]

        info_group = QGroupBox("Información")
        info_layout = QVBoxLayout()
        info_text = (
            f"<b>Canales:</b> {', '.join(channels)}<br>"
            f"<b>Dimensiones:</b> {self.model_data['combined'].shape}<br>"
            f"<b>Tipo de dato:</b> {self.model_data['combined'].dtype}<br><br>"
            f"<b>Descripción:</b><br>{self.get_model_description()}"
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background-color: #f5f5f5; border-radius: 3px;")
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        stats_group = QGroupBox("Estadísticas por canal")
        stats_layout = QVBoxLayout()
        for ch in channels:
            lbl = QLabel(self.get_channel_stats(self.model_data[ch], ch))
            lbl.setStyleSheet(f"""
                padding: 8px;
                background-color: {self.get_channel_color(ch)};
                border-radius: 3px; font-family: monospace;
            """)
            stats_layout.addWidget(lbl)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        btn_save = QPushButton("Guardar modelo completo")
        btn_save.clicked.connect(self.save_model)
        layout.addWidget(btn_save)
        layout.addStretch()
        return panel

    def create_channels_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        tabs = QTabWidget()
        tabs.addTab(self.create_channels_view(),    "Canales por separado")
        tabs.addTab(self.create_combined_view(),    "Modelo completo")
        tabs.addTab(self.create_comparison_view(), "Comparación")
        layout.addWidget(tabs)
        return panel

    def create_channels_view(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        for ch in (k for k in self.model_data if k != "combined"):
            layout.addWidget(self.create_channel_display(self.model_data[ch], ch.upper()))
        return widget

    def create_channel_display(self, channel_data, title):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: 14px; font-weight: bold; padding: 5px;
            background-color: {self.get_channel_color(title)}; border-radius: 3px;
        """)
        layout.addWidget(title_label)

        viewer = QLabel()
        viewer.setAlignment(Qt.AlignCenter)
        viewer.setMinimumSize(300, 300)
        viewer.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.display_image(viewer, self.prepare_channel_for_display(channel_data, title))
        layout.addWidget(viewer)
        return widget

    def create_combined_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        viewer = QLabel()
        viewer.setAlignment(Qt.AlignCenter)
        viewer.setMinimumSize(600, 400)
        viewer.setStyleSheet("border: 2px solid gray; background-color: #2b2b2b;")
        self.display_image(viewer, self.prepare_combined_for_display())
        layout.addWidget(viewer)
        return widget

    def create_comparison_view(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        for label_text, img in [("Original", self.original_image),
                                 (self.model_name, self.prepare_combined_for_display())]:
            sub = QWidget()
            sub_layout = QVBoxLayout(sub)
            sub_layout.addWidget(QLabel(label_text))
            viewer = QLabel()
            viewer.setAlignment(Qt.AlignCenter)
            viewer.setMinimumSize(400, 400)
            viewer.setStyleSheet("border: 1px solid gray;")
            self.display_image(viewer, img)
            sub_layout.addWidget(viewer)
            layout.addWidget(sub)

        return widget

    # ------------------------------------------------------------------
    # Preparación de imágenes para mostrar
    # ------------------------------------------------------------------

    def prepare_channel_for_display(self, channel_data, channel_name):
        if channel_data is None:
            return None
        if channel_data.dtype != np.uint8:
            channel_norm = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            channel_norm = channel_data
        h, w = channel_norm.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        cn = channel_name.lower()
        if cn in ("r", "red", "rojo"):
            rgb_img[:, :, 0] = channel_norm
        elif cn in ("g", "green", "verde"):
            rgb_img[:, :, 1] = channel_norm
        elif cn in ("b", "blue", "azul"):
            rgb_img[:, :, 2] = channel_norm
        elif cn in ("c", "cyan"):
            rgb_img[:, :, 1] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        elif cn in ("m", "magenta"):
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        elif cn in ("y", "yellow"):
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 1] = channel_norm
        elif cn in ("k", "key", "black"):
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 1] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        elif cn in ("h", "hue"):
            h_color = cv2.applyColorMap(channel_norm, cv2.COLORMAP_HSV)
            return cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        else:
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 1] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        return rgb_img

    def prepare_combined_for_display(self):
        combined = self.model_data["combined"]
        conversions = {
            "HSV": cv2.COLOR_HSV2RGB,
            "HSL": cv2.COLOR_HLS2RGB,
            "LAB": cv2.COLOR_LAB2RGB,
            "YUV": cv2.COLOR_YUV2RGB,
            "XYZ": cv2.COLOR_XYZ2RGB,
        }
        if self.model_name in conversions:
            return cv2.cvtColor(combined, conversions[self.model_name])
        if self.model_name == "CMYK":
            return self.cmyk_to_rgb_display()
        return combined  # RGB: ya está en formato correcto

    def cmyk_to_rgb_display(self):
        """Convierte CMYK a RGB para visualización.

        BUG FIX: La versión anterior no clampaba K, produciendo valores
        negativos en R/G/B cuando k > 1.0 por errores de punto flotante.
        Ahora todos los canales se clampan a [0, 1] antes de escalar.
        """
        c = self.model_data["c"].astype(np.float32) / 255.0
        m = self.model_data["m"].astype(np.float32) / 255.0
        y = self.model_data["y"].astype(np.float32) / 255.0
        k = self.model_data["k"].astype(np.float32) / 255.0

        # Clamp para evitar valores fuera de rango por errores de fp
        k = np.clip(k, 0.0, 1.0)

        r = np.clip((1.0 - c) * (1.0 - k), 0.0, 1.0) * 255
        g = np.clip((1.0 - m) * (1.0 - k), 0.0, 1.0) * 255
        b = np.clip((1.0 - y) * (1.0 - k), 0.0, 1.0) * 255

        return cv2.merge([
            r.astype(np.uint8),
            g.astype(np.uint8),
            b.astype(np.uint8),
        ])

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def display_image(self, viewer, img):
        if img is None:
            return
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            q_img = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        viewer.setPixmap(
            pixmap.scaled(viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def get_model_description(self):
        descriptions = {
            "RGB":  "Modelo aditivo basado en Rojo, Verde y Azul",
            "CMYK": "Modelo sustractivo usado en impresión: Cyan, Magenta, Yellow, Key(Black)",
            "HSV":  "Modelo perceptual: Hue (Matiz), Saturation (Saturación), Value (Valor)",
            "HSI":  "Modelo basado en percepción humana: Hue, Saturation, Intensity",
            "YUV":  "Modelo usado en TV: Luminancia (Y) y Crominancia (U, V)",
            "LAB":  "Modelo perceptual uniforme CIE L*a*b*",
            "XYZ":  "Modelo de color CIE 1931 (espacio de color estándar)",
        }
        return descriptions.get(self.model_name, "Sin descripción disponible")

    def get_channel_stats(self, channel_data, channel_name):
        if channel_data is None:
            return f"{channel_name}: Sin datos"
        return (
            f"{channel_name}: "
            f"Min={np.min(channel_data)}, "
            f"Max={np.max(channel_data)}, "
            f"Media={np.mean(channel_data):.1f}, "
            f"Std={np.std(channel_data):.1f}"
        )

    def get_channel_color(self, channel_name):
        colors = {
            "R": "#FFE5E5", "G": "#E5FFE5", "B": "#E5E5FF",
            "C": "#E5FFFF", "M": "#FFE5FF", "Y": "#FFFFE5", "K": "#E5E5E5",
            "H": "#FFE5CC", "S": "#CCFFE5", "V": "#E5CCFF",
            "I": "#FFCCE5", "U": "#CCE5FF",
            "L": "#FFFFFF", "A": "#FFCCCC",
            "X": "#E5CCCC", "Z": "#CCCCE5",
        }
        return colors.get(channel_name.upper(), "#F0F0F0")

    def save_model(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Guardar modelo",
            f"modelo_{self.model_name.lower()}.jpg",
            "JPEG (*.jpg);;PNG (*.png)",
        )
        if filepath:
            img_to_save = self.prepare_combined_for_display()
            cv2.imwrite(filepath, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
            self.statusBar().showMessage(f"Modelo guardado en: {filepath}")
