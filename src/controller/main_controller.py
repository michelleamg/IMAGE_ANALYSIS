from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2

class MainController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Abrir imagen",
            "",
            "Imagenes (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Todos (*.*)"
        )
        if not path:
            return

        img = cv2.imread(path)  # BGR
        if img is None:
            QMessageBox.critical(self.view, "Error", "No se pudo abrir la imagen.")
            return

        self.model.set_image(path, img)
        self.view.set_status(f"Imagen cargada: {path}")
        self.view.show_image_bgr(img)