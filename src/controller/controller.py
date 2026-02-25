from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os

class ImageController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.current_image = None
        self.current_map = None
    
    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self.view, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)" 
        )
        
        if filepath:
            try:
                name = self.model.load_image(filepath)
                self.current_image = name
                
                img = self.model.get_image(name)
                self.view.show_original(img)
                self.view.show_status(f"Imagen cargada")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))
    
    def apply_map(self):
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        map_name = self.view.map_combo.currentText()
        
        if map_name == "PERSONALIZADO":
            result = self.model.apply_personalized_map(self.current_image)
            self.current_map = 'Personalizado'
            map_display = "Personalizado"
        else:
            result = self.model.apply_colormap(self.current_image, map_name)
            self.current_map = map_name
            map_display = map_name
        
        img_rgb = self.model.get_result(self.current_image, self.current_map)
        self.view.show_result(img_rgb)
        self.view.show_status(f"Mapa {map_display} aplicado")
    
    def save_result(self):
        if not self.current_image or not self.current_map:
            QMessageBox.warning(self.view, "Aviso", "No hay resultado para guardar")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self.view, "Guardar imagen", 
            f"{self.current_image}_{self.current_map}.jpg",
            "JPEG (*.jpg);;PNG (*.png)"
        )
        
        if filepath:
            cv_img = self.model.get_result(self.current_image, self.current_map)
            if cv_img is not None:
                import cv2
                cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, cv_img_bgr)
                self.view.show_status(f"Guardado: {os.path.basename(filepath)}")