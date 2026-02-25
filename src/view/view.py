from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b;")
        self.setText("No image")
    
    def set_image(self, cv_img):
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.init_ui()
    
    def set_controller(self, controller):
        self.controller = controller
        self.connect_signals()
    
    def init_ui(self):
        self.setWindowTitle("Práctica 1: Mapas de Color")
        self.setGeometry(100, 100, 1000, 600)
        
        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Panel izquierdo (controles)
        left = QWidget()
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)
        
        # 1. Cargar imagen
        load_group = QGroupBox("1. Cargar imagen")
        load_layout = QVBoxLayout()
        self.btn_load = QPushButton("📁 Seleccionar imagen")
        load_layout.addWidget(self.btn_load)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)
        
        # 2. Mapas de color 
        maps_group = QGroupBox("2. Mapas de Color")
        maps_layout = QVBoxLayout()
        
        self.map_combo = QComboBox()
        self.map_combo.addItems([
            "JET", "HOT", "COOL", "SPRING", 
            "SUMMER", "WINTER", "HSV", "PARULA",
            "PERSONALIZADO"
        ])
        maps_layout.addWidget(QLabel("Selecciona un mapa:"))
        maps_layout.addWidget(self.map_combo)
        
        # Botón aplicar mapa (¡ESTE FALTABA!)
        self.btn_apply = QPushButton("🎨 Aplicar mapa")
        maps_layout.addWidget(self.btn_apply)
        
        maps_group.setLayout(maps_layout)
        left_layout.addWidget(maps_group)
        
        # 3. Guardar imagen
        save_group = QGroupBox("3. Guardar")
        save_layout = QVBoxLayout()
        self.btn_save = QPushButton("Guardar imagen")
        save_layout.addWidget(self.btn_save)
        save_group.setLayout(save_layout)
        left_layout.addWidget(save_group)
        
        left_layout.addStretch()
        
        # Panel derecho (visualización)
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
        self.status_label = QLabel("📌 Listo")
        self.status_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        right_layout.addWidget(self.status_label)
        
        # Añadir paneles
        layout.addWidget(left)
        layout.addWidget(right, 1)
    
    def connect_signals(self):
        self.btn_load.clicked.connect(self.controller.load_image)
        self.btn_apply.clicked.connect(self.controller.apply_map)
        self.btn_save.clicked.connect(self.controller.save_result)
    
    def show_original(self, img):
        self.original_view.set_image(img)
    
    def show_result(self, img):
        self.result_view.set_image(img)
        self.tabs.setCurrentIndex(1)
    
    def show_status(self, msg):
        self.status_label.setText(f"{msg}")