"""Ventana para mostrar componentes RGB por separado"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np

class RGBComponentsWindow(QMainWindow):
    def __init__(self, r_component, g_component, b_component, image_name):
        super().__init__()
        self.setWindowTitle(f"Componentes RGB - {image_name}")
        self.setGeometry(200, 200, 1200, 400)
        
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Crear vistas para cada componente
        self.view_r = ComponentViewer(r_component, 'Rojos', '#FFE5E5')
        self.view_g = ComponentViewer(g_component, 'Verdes', '#E5FFE5')
        self.view_b = ComponentViewer(b_component, 'Azules', '#E5E5FF')
        
        # Añadir al layout
        layout.addWidget(self.view_r)
        layout.addWidget(self.view_g)
        layout.addWidget(self.view_b)
        
        # Barra de estado
        self.statusBar().showMessage("Componentes RGB mostrados en ventanas separadas")

class ComponentViewer(QWidget):
    def __init__(self, component_data, title, bg_color):
        super().__init__()
        self.component_data = component_data
        self.title = title
        self.bg_color = bg_color
        
        layout = QVBoxLayout(self)
        
        # Título
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            padding: 10px;
            background-color: {bg_color};
            border-radius: 5px;
        """)
        layout.addWidget(title_label)
        
        # Visor de imagen
        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setMinimumSize(350, 350)
        self.image_view.setStyleSheet("border: 2px solid gray; background-color: #2b2b2b;")
        layout.addWidget(self.image_view)
        
        # Información del componente
        info_label = QLabel(self.get_component_info())
        info_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        layout.addWidget(info_label)
        
        # Mostrar la imagen
        self.display_component()
    
    def get_component_info(self):
        """Obtiene información estadística del componente"""
        if self.component_data is None:
            return "Sin datos"
        
        min_val = np.min(self.component_data)
        max_val = np.max(self.component_data)
        mean_val = np.mean(self.component_data)
        
        return f"Min: {min_val} | Max: {max_val} | Media: {mean_val:.1f}"
    
    def display_component(self):
        """Convierte y muestra el componente en la vista"""
        if self.component_data is None:
            return
        
        # Normalizar para visualización
        if self.component_data.dtype != np.uint8:
            component_norm = cv2.normalize(self.component_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            component_norm = self.component_data
        
        # Crear imagen RGB para visualización
        h, w = component_norm.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.title == 'Rojos':
            rgb_image[:, :, 0] = component_norm  # Canal R
        elif self.title == 'Verdes':
            rgb_image[:, :, 1] = component_norm  # Canal G
        elif self.title == 'Azules':
            rgb_image[:, :, 2] = component_norm  # Canal B
        
        # Convertir a QPixmap
        bytes_per_line = 3 * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Escalar manteniendo aspecto
        scaled = pixmap.scaled(self.image_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_view.setPixmap(scaled)
    
    def resizeEvent(self, event):
        """Redimensiona la imagen cuando cambia el tamaño del widget"""
        super().resizeEvent(event)
        if hasattr(self, 'component_data') and self.component_data is not None:
            self.display_component()