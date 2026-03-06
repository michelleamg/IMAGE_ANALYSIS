"""Ventana para mostrar modelos de color"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
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
        
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Panel izquierdo (información)
        left_panel = self.create_info_panel()
        layout.addWidget(left_panel, 1)
        
        # Panel derecho (canales)
        right_panel = self.create_channels_panel()
        layout.addWidget(right_panel, 3)
        
        self.show_status()
    
    def create_info_panel(self):
        """Crea panel con información del modelo"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        
        # Título
        title = QLabel(f"Modelo: {self.model_name}")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Información general
        info_group = QGroupBox("Información")
        info_layout = QVBoxLayout()
        
        channels = list(self.model_data.keys())
        channels.remove('combined')
        
        info_text = f"""
        <b>Canales:</b> {', '.join(channels)}<br>
        <b>Dimensiones:</b> {self.model_data['combined'].shape}<br>
        <b>Tipo de dato:</b> {self.model_data['combined'].dtype}<br>
        <br>
        <b>Descripción:</b><br>
        {self.get_model_description()}
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background-color: #f5f5f5; border-radius: 3px;")
        info_layout.addWidget(info_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Estadísticas por canal
        stats_group = QGroupBox("Estadísticas por canal")
        stats_layout = QVBoxLayout()
        
        for channel_name in channels:
            channel_data = self.model_data[channel_name]
            stats = self.get_channel_stats(channel_data, channel_name)
            
            channel_stats = QLabel(stats)
            channel_stats.setStyleSheet(f"""
                padding: 8px;
                background-color: {self.get_channel_color(channel_name)};
                border-radius: 3px;
                font-family: monospace;
            """)
            stats_layout.addWidget(channel_stats)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Botón para guardar
        btn_save = QPushButton("Guardar modelo completo")
        btn_save.clicked.connect(self.save_model)
        layout.addWidget(btn_save)
        
        layout.addStretch()
        
        return panel
    
    def create_channels_panel(self):
        """Crea panel con visualización de canales"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs para diferentes visualizaciones
        tabs = QTabWidget()
        
        # Tab 1: Canales por separado
        channels_widget = self.create_channels_view()
        tabs.addTab(channels_widget, "Canales por separado")
        
        # Tab 2: Modelo completo
        combined_widget = self.create_combined_view()
        tabs.addTab(combined_widget, "Modelo completo")
        
        # Tab 3: Comparación con original
        compare_widget = self.create_comparison_view()
        tabs.addTab(compare_widget, "Comparación")
        
        layout.addWidget(tabs)
        
        return panel
    
    def create_channels_view(self):
        """Crea vista de canales individuales"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        channels = [k for k in self.model_data.keys() if k != 'combined']
        
        for channel_name in channels:
            channel_widget = self.create_channel_display(
                self.model_data[channel_name],
                channel_name.upper()
            )
            layout.addWidget(channel_widget)
        
        return widget
    
    def create_channel_display(self, channel_data, title):
        """Crea un widget para mostrar un canal"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Título del canal
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            padding: 5px;
            background-color: {self.get_channel_color(title)};
            border-radius: 3px;
        """)
        layout.addWidget(title_label)
        
        # Visor de imagen
        viewer = QLabel()
        viewer.setAlignment(Qt.AlignCenter)
        viewer.setMinimumSize(300, 300)
        viewer.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        
        # Convertir a imagen RGB para visualización
        display_img = self.prepare_channel_for_display(channel_data, title)
        self.display_image(viewer, display_img)
        
        layout.addWidget(viewer)
        
        return widget
    
    def create_combined_view(self):
        """Crea vista del modelo combinado"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Visor principal
        viewer = QLabel()
        viewer.setAlignment(Qt.AlignCenter)
        viewer.setMinimumSize(600, 400)
        viewer.setStyleSheet("border: 2px solid gray; background-color: #2b2b2b;")
        
        # Convertir a RGB para visualización
        combined_img = self.prepare_combined_for_display()
        self.display_image(viewer, combined_img)
        
        layout.addWidget(viewer)
        
        return widget
    
    def create_comparison_view(self):
        """Crea vista de comparación con original"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Original
        orig_widget = QWidget()
        orig_layout = QVBoxLayout(orig_widget)
        orig_layout.addWidget(QLabel("Original"))
        orig_viewer = QLabel()
        orig_viewer.setAlignment(Qt.AlignCenter)
        orig_viewer.setMinimumSize(400, 400)
        orig_viewer.setStyleSheet("border: 1px solid gray;")
        self.display_image(orig_viewer, self.original_image)
        orig_layout.addWidget(orig_viewer)
        
        # Modelo
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        model_layout.addWidget(QLabel(self.model_name))
        model_viewer = QLabel()
        model_viewer.setAlignment(Qt.AlignCenter)
        model_viewer.setMinimumSize(400, 400)
        model_viewer.setStyleSheet("border: 1px solid gray;")
        model_img = self.prepare_combined_for_display()
        self.display_image(model_viewer, model_img)
        model_layout.addWidget(model_viewer)
        
        layout.addWidget(orig_widget)
        layout.addWidget(model_widget)
        
        return widget
    
    def prepare_channel_for_display(self, channel_data, channel_name):
        """Prepara un canal para visualización en color"""
        if channel_data is None:
            return None
        
        h, w = channel_data.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Normalizar si es necesario
        if channel_data.dtype != np.uint8:
            channel_norm = cv2.normalize(channel_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            channel_norm = channel_data
        
        # Asignar color según el canal
        if channel_name.lower() in ['r', 'red', 'rojo']:
            rgb_img[:, :, 0] = channel_norm
        elif channel_name.lower() in ['g', 'green', 'verde']:
            rgb_img[:, :, 1] = channel_norm
        elif channel_name.lower() in ['b', 'blue', 'azul']:
            rgb_img[:, :, 2] = channel_norm
        elif channel_name.lower() in ['c', 'cyan']:
            rgb_img[:, :, 1] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        elif channel_name.lower() in ['m', 'magenta']:
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        elif channel_name.lower() in ['y', 'yellow']:
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 1] = channel_norm
        elif channel_name.lower() in ['k', 'key', 'black']:
            # Negro: todos los canales iguales
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 1] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        elif channel_name.lower() in ['h', 'hue']:
            # Hue: usar colormap
            h_norm = cv2.applyColorMap(channel_norm, cv2.COLORMAP_HSV)
            return cv2.cvtColor(h_norm, cv2.COLOR_BGR2RGB)
        else:
            # Por defecto: escala de grises
            rgb_img[:, :, 0] = channel_norm
            rgb_img[:, :, 1] = channel_norm
            rgb_img[:, :, 2] = channel_norm
        
        return rgb_img
    
    def prepare_combined_for_display(self):
        """Prepara el modelo combinado para visualización"""
        combined = self.model_data['combined']
        
        # Convertir a RGB según el modelo
        if self.model_name in ['HSV', 'HSL']:
            return cv2.cvtColor(combined, cv2.COLOR_HSV2RGB)
        elif self.model_name == 'LAB':
            return cv2.cvtColor(combined, cv2.COLOR_LAB2RGB)
        elif self.model_name == 'YUV':
            return cv2.cvtColor(combined, cv2.COLOR_YUV2RGB)
        elif self.model_name == 'XYZ':
            return cv2.cvtColor(combined, cv2.COLOR_XYZ2RGB)
        elif self.model_name == 'CMYK':
            # CMYK necesita conversión especial
            return self.cmyk_to_rgb_display()
        else:
            # Asumir que ya es RGB
            return combined
    
    def cmyk_to_rgb_display(self):
        """Convierte CMYK a RGB para visualización"""
        c = self.model_data['c'].astype(np.float32) / 255.0
        m = self.model_data['m'].astype(np.float32) / 255.0
        y = self.model_data['y'].astype(np.float32) / 255.0
        k = self.model_data['k'].astype(np.float32) / 255.0
        
        r = (1 - c) * (1 - k) * 255
        g = (1 - m) * (1 - k) * 255
        b = (1 - y) * (1 - k) * 255
        
        return cv2.merge([
            r.astype(np.uint8),
            g.astype(np.uint8),
            b.astype(np.uint8)
        ])
    
    def display_image(self, viewer, img):
        """Muestra una imagen en el visor"""
        if img is None:
            return
        
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            bytes_per_line = w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * w
            q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        viewer.setPixmap(scaled)
    
    def get_model_description(self):
        """Obtiene descripción del modelo de color"""
        descriptions = {
            'RGB': 'Modelo aditivo basado en Rojo, Verde y Azul',
            'CMYK': 'Modelo sustractivo usado en impresión: Cyan, Magenta, Yellow, Key(Black)',
            'HSV': 'Modelo perceptual: Hue (Matiz), Saturation (Saturación), Value (Valor)',
            'HSI': 'Modelo basado en percepción humana: Hue, Saturation, Intensity',
            'YUV': 'Modelo usado en TV: Luminancia (Y) y Crominancia (U,V)',
            'LAB': 'Modelo perceptual uniforme CIE L*a*b*',
            'XYZ': 'Modelo de color CIE 1931 (espacio de color estándar)'
        }
        return descriptions.get(self.model_name, 'Sin descripción disponible')
    
    def get_channel_stats(self, channel_data, channel_name):
        """Calcula estadísticas para un canal"""
        if channel_data is None:
            return f"{channel_name}: Sin datos"
        
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        
        return f"{channel_name}: Min={min_val}, Max={max_val}, Media={mean_val:.1f}, Std={std_val:.1f}"
    
    def get_channel_color(self, channel_name):
        """Obtiene color de fondo para cada canal"""
        colors = {
            'R': '#FFE5E5', 'G': '#E5FFE5', 'B': '#E5E5FF',
            'C': '#E5FFFF', 'M': '#FFE5FF', 'Y': '#FFFFE5', 'K': '#E5E5E5',
            'H': '#FFE5CC', 'S': '#CCFFE5', 'V': '#E5CCFF',
            'I': '#FFCCE5', 'U': '#CCE5FF', 'V': '#E5FFCC',
            'L': '#FFFFFF', 'A': '#FFCCCC', 'B': '#CCCCFF',
            'X': '#E5CCCC', 'Y': '#CCE5CC', 'Z': '#CCCCE5'
        }
        return colors.get(channel_name.upper(), '#F0F0F0')
    
    def show_status(self):
        """Muestra mensaje en barra de estado"""
        self.statusBar().showMessage(f"Modelo {self.model_name} cargado correctamente")
    
    def save_model(self):
        """Guarda el modelo combinado"""
        from PyQt5.QtWidgets import QFileDialog
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Guardar modelo",
            f"modelo_{self.model_name.lower()}.jpg",
            "JPEG (*.jpg);;PNG (*.png)"
        )
        
        if filepath:
            img_to_save = self.prepare_combined_for_display()
            img_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, img_bgr)
            self.statusBar().showMessage(f"Modelo guardado en: {filepath}")