"""Vista del programa, define la interfaz gráfica y los widgets.
- Autor: Alejandra Michelle Mateo Garcia & Leyva Triana Isis Valeria
- Fecha: 20 de febrero del 2026
- Versión: 2.2
- Descripción: Práctica 1 - "Explorando la Imagen Digital con Python"
               Implementación completa incluyendo:
               ✓ Lectura de imágenes (OpenCV)
               ✓ Separación de componentes RGB
               ✓ Escalamiento a grises
               ✓ Mapas de color (OpenCV + personalizado morado-fucsia)
               ✓ Binarización (fijo, Otsu, adaptativo)
               ✓ Histogramas de intensidad
               ✓ Interfaz gráfica con PyQt5 (MVC)
- Escuela: ESCOM-IPN    
- Materia: Análisis de Imágenes
Librerias utilizadas:
- PyQt5: Para la construcción de la interfaz gráfica (widgets, layouts, señales)
- Matplotlib: Para la visualización de histogramas dentro de la aplicación
- OpenCV: Para la manipulación y procesamiento de imágenes (convertir a QImage,
    aplicar mapas de color, etc.)
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        # Configura etiqueta para mostrar imágenes centradas con fondo oscuro
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.setText("No image")
    
    def set_image(self, cv_img):
        # Convierte un arreglo RGB de OpenCV en QPixmap ajustado al widget
        if cv_img is None:
            self.clear()
            self.setText("No image")
            return
        
        h, w, c = cv_img.shape
        bytes_per_line = 3 * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

class MplCanvas(FigureCanvas):
    """Canvas para matplotlib integrado en Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Crea una figura de Matplotlib incrustada en el canvas Qt
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = None
        # Construye toda la interfaz principal al arrancar
        self.init_ui()
    
    def set_controller(self, controller):
        # Asigna el controlador y engancha las señales de la UI
        self.controller = controller
        self.connect_signals()
    
    def init_ui(self):
        # Define layout, pestañas y controles de la aplicación
        self.setWindowTitle("Práctica 1: Explorando la Imagen Digital con Python")
        self.setGeometry(100, 100, 1200, 700)
        
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Panel izquierdo (controles)
        left = QWidget()
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)
        
        # ===== 1. Cargar imagen =====
        load_group = QGroupBox("1. Cargar imagen")
        load_layout = QVBoxLayout()
        self.btn_load = QPushButton("Seleccionar imagen")
        load_layout.addWidget(self.btn_load)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)
        
        # ===== 2. Componentes RGB =====
        rgb_group = QGroupBox("2. Componentes RGB")
        rgb_layout = QVBoxLayout()
        self.btn_rgb = QPushButton("Mostrar componentes RGB")
        rgb_layout.addWidget(self.btn_rgb)
        rgb_group.setLayout(rgb_layout)
        left_layout.addWidget(rgb_group)

        models_group = QGroupBox("3. Modelos de Color")
        models_layout = QVBoxLayout()

        self.models_combo = QComboBox()
        self.models_combo.addItems([
            "RGB (Original)",
            "CMYK (Cian, Magenta, Amarillo, Negro)",
            "HSV (Matiz, Saturación, Valor)",
            "HSI (Matiz, Saturación, Intensidad)",
            "YUV (Luminancia, Crominancia)",
            "LAB (CIE L*a*b*)",
            "XYZ (CIE 1931)"
        ])
        models_layout.addWidget(QLabel("Selecciona modelo:"))
        models_layout.addWidget(self.models_combo)

        self.btn_apply_model = QPushButton("Aplicar modelo de color")
        models_layout.addWidget(self.btn_apply_model)

        models_group.setLayout(models_layout)
        left_layout.addWidget(models_group)
        
        # ===== 4. Escala de grises =====
        gray_group = QGroupBox("4. Escala de grises")
        gray_layout = QVBoxLayout()
        self.btn_gray = QPushButton("Convertir a grises")
        gray_layout.addWidget(self.btn_gray)
        gray_group.setLayout(gray_layout)
        left_layout.addWidget(gray_group)
        
        # ===== 4. Mapas de color =====
        maps_group = QGroupBox("5. Mapas de Color")
        maps_layout = QVBoxLayout()
        
        self.map_combo = QComboBox()
        self.map_combo.addItems([
            "TWILIGHT", "TURBO", "VIRDIS", "PINK", 
            "INFERNO", "WINTER", "HSV", "PARULA",
            "PERSONALIZADO (Morado-Fucsia)"
        ])
        maps_layout.addWidget(QLabel("Selecciona un mapa:"))
        maps_layout.addWidget(self.map_combo)
        
        self.btn_apply_map = QPushButton("Aplicar mapa")
        maps_layout.addWidget(self.btn_apply_map)
        
        maps_group.setLayout(maps_layout)
        left_layout.addWidget(maps_group)
        
        # ===== 5. Binarización =====
        binary_group = QGroupBox("6. Binarización")
        binary_layout = QVBoxLayout()
        
        # Slider para umbral
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Umbral:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(128)
        self.threshold_label = QLabel("128")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        binary_layout.addLayout(threshold_layout)
        
        # Botones de binarización
        btn_layout = QHBoxLayout()
        self.btn_binary_fixed = QPushButton("Fijo")
        self.btn_binary_otsu = QPushButton("Otsu")
        self.btn_binary_adaptive = QPushButton("Adaptativo")
        btn_layout.addWidget(self.btn_binary_fixed)
        btn_layout.addWidget(self.btn_binary_otsu)
        btn_layout.addWidget(self.btn_binary_adaptive)
        binary_layout.addLayout(btn_layout)
        
        binary_group.setLayout(binary_layout)
        left_layout.addWidget(binary_group)
        
        # ===== 6. Histograma =====
        hist_group = QGroupBox("7. Histograma")
        hist_layout = QVBoxLayout()
        self.btn_hist = QPushButton("Mostrar histograma")
        hist_layout.addWidget(self.btn_hist)
        hist_group.setLayout(hist_layout)
        left_layout.addWidget(hist_group)

        
        # ===== 8. Guardar =====
        save_group = QGroupBox("8. Guardar")
        save_layout = QVBoxLayout()
        self.btn_save = QPushButton("Guardar resultado")
        save_layout.addWidget(self.btn_save)
        save_group.setLayout(save_layout)
        left_layout.addWidget(save_group)
        
        left_layout.addStretch()
        
        # ===== Panel derecho =====
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        # Tabs
        self.tabs = QTabWidget()
        self.original_view = ImageViewer()
        self.result_view = ImageViewer()
        
        self.tabs.addTab(self.original_view, "Original")
        self.tabs.addTab(self.result_view, "Resultado")
        
        right_layout.addWidget(self.tabs)
        
        # Barra de estado
        self.status_label = QLabel("Cargar una imagen para comenzar")
        self.status_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        right_layout.addWidget(self.status_label)
        
        # Añadir paneles
        layout.addWidget(left)
        layout.addWidget(right, 1)
        
        # Conectar slider
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
    
    def update_threshold_label(self, value):
        # Refleja el valor del umbral en la etiqueta numérica
        self.threshold_label.setText(str(value))
    
    def connect_signals(self):
        # Enlaza botones y sliders con acciones del controlador
        self.btn_load.clicked.connect(self.controller.load_image)
        self.btn_rgb.clicked.connect(self.controller.show_rgb_components)
        self.btn_apply_model.clicked.connect(self.controller.apply_color_model)  # NUEVO
        self.btn_gray.clicked.connect(self.controller.convert_to_gray)
        self.btn_apply_map.clicked.connect(self.controller.apply_map)
        self.btn_binary_fixed.clicked.connect(lambda: self.controller.apply_binary('fixed'))
        self.btn_binary_otsu.clicked.connect(lambda: self.controller.apply_binary('otsu'))
        self.btn_binary_adaptive.clicked.connect(lambda: self.controller.apply_binary('adaptive'))
        self.btn_hist.clicked.connect(self.controller.show_histogram)
        self.btn_save.clicked.connect(self.controller.save_result)

    def show_original(self, img):
        # Muestra la imagen original cargada en su pestaña
        self.original_view.set_image(img)
    
    def show_result(self, img):
        # Coloca el resultado procesado y activa la pestaña de resultado
        self.result_view.set_image(img)
        self.tabs.setCurrentIndex(1)
    
    def show_status(self, msg):
        # Actualiza la barra de estado con un mensaje breve
        self.status_label.setText(f"📌 {msg}")
