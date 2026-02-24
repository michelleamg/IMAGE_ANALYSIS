from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QStatusBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2

class MainView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDI App - PyQt5 (MVC Base)")
        self.resize(1000, 650)

        self.controller = None

        # --- UI base ---
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        main_layout.addLayout(top_bar)

        self.btn_open = QPushButton("Abrir imagen")
        self.btn_open.clicked.connect(self._on_open)
        top_bar.addWidget(self.btn_open)

        top_bar.addStretch(1)

        self.image_label = QLabel("Aquí se mostrará la imagen")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #111; color: #ddd; border-radius: 8px; padding: 10px;")
        self.image_label.setMinimumHeight(450)
        main_layout.addWidget(self.image_label, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.set_status("Listo.")

    def set_controller(self, controller):
        self.controller = controller

    def _on_open(self):
        if self.controller:
            self.controller.open_image()

    def set_status(self, text: str):
        self.status.showMessage(text)

    def show_image_bgr(self, img_bgr):
        """
        Recibe imagen OpenCV (BGR) y la muestra escalada al label.
        """
        if img_bgr is None:
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        bytes_per_line = 3 * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Escalar manteniendo proporción
        pix = pix.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    def resizeEvent(self, event):
        """
        Para que cuando redimensiones la ventana, la imagen se re-escale.
        """
        super().resizeEvent(event)
        # Si ya hay imagen puesta, re-escalar el pixmap actual
        pix = self.image_label.pixmap()
        if pix:
            self.image_label.setPixmap(
                pix.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )