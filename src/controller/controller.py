"""Controlador del programa, maneja la lógica de interacción entre el modelo y la vista.
- Autor: Alejandra Michelle Mateo Garcia & Leyva Triana Isis Valeria
- Fecha: 20 de febrero del 2026
- Versión: 2.0
- Descripción: Práctica 1 - "Explorando la Imagen Digital con Python
-liberias utilizadas:
- PyQt5: Para la construcción de la interfaz gráfica (widgets, layouts, señales)    
- Matplotlib: Para la visualización de histogramas dentro de la aplicación
- OpenCV: Para la manipulación y procesamiento de imágenes (convertir a QImage,
    aplicar mapas de color, etc.)
- Escuela: ESCOM-IPN
- Materia: Análisis de Imágenes
"""
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

class ImageController:
    def __init__(self, model, view):
        # Configura el vínculo MVC y estado base de la sesión
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.current_image = None
        self.current_map = None
    
    def load_image(self):
        # Abre un diálogo de archivo y registra la imagen seleccionada
        filepath, _ = QFileDialog.getOpenFileName(
            self.view, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if filepath:
            try:
                name = self.model.load_image(filepath)
                self.current_image = name
                
                # Mostrar imagen original a color
                img = self.model.get_image(name)
                self.view.show_original(img)
                self.view.show_status(f"Imagen a color cargada: {name}")
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))
    
    def convert_to_gray(self):
        # Convierte la imagen cargada a escala de grises y la muestra
        """Convierte la imagen a color a escala de grises"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        # Convertir a grises
        self.model.convert_to_gray(self.current_image)
        
        # Mostrar resultado en grises
        img_gray = self.model.get_gray_image(self.current_image)
        if img_gray is not None:
            self.view.show_result(img_gray)
            self.view.show_status("Imagen convertida a escala de grises")
    
    def show_rgb_components(self):
        # Visualiza cada canal R, G y B en un subplot
        """Muestra los componentes RGB por separado"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        r, g, b = self.model.get_rgb_components(self.current_image)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Componentes RGB - {self.current_image}", fontsize=14)
        
        axes[0].imshow(r, cmap='Reds')
        axes[0].set_title('Componente Rojo (R)')
        axes[0].axis('off')
        
        axes[1].imshow(g, cmap='Greens')
        axes[1].set_title('Componente Verde (G)')
        axes[1].axis('off')
        
        axes[2].imshow(b, cmap='Blues')
        axes[2].set_title('Componente Azul (B)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        self.view.show_status("Componentes RGB mostrados")
    
    def apply_map(self):
        # Aplica el mapa de color elegido (OpenCV o personalizado)
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        map_name = self.view.map_combo.currentText()
        
        if "PERSONALIZADO" in map_name:
            result = self.model.apply_personalized_map(self.current_image)
            self.current_map = 'Personalizado'
            map_display = "Personalizado (Morado-Fucsia)"
        else:
            result = self.model.apply_colormap(self.current_image, map_name)
            self.current_map = map_name
            map_display = map_name
        
        img_rgb = self.model.get_result(self.current_image, self.current_map)
        if img_rgb is not None:
            self.view.show_result(img_rgb)
            self.view.show_status(f"Mapa {map_display} aplicado")
    
    def apply_binary(self, method):
        # Ejecuta binarización con el método seleccionado
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        threshold = self.view.threshold_slider.value()
        result = self.model.apply_threshold(self.current_image, threshold, method)
        self.current_map = f'BINARIA_{method}'
        
        img_rgb = self.model.get_result(self.current_image, self.current_map)
        if img_rgb is not None:
            self.view.show_result(img_rgb)
            self.view.show_status(f"Binarización {method} aplicada")
    
    def save_result(self):
        # Exporta a disco el resultado actualmente mostrado
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
                cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, cv_img_bgr)
                self.view.show_status(f"Guardado: {os.path.basename(filepath)}")

    def show_histogram(self):
        # Muestra histograma de la imagen según pestaña y mapa activo
        """Muestra el histograma según el tipo de imagen actual"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        # Verificar qué pestaña está activa
        current_tab = self.view.tabs.currentIndex()
        
        if current_tab == 0:  # Pestaña Original
            # Mostrar histograma RGB de la imagen original a color
            hists = self.model.get_histogram_rgb(self.current_image)
            img = self.model.get_image(self.current_image)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"Histograma RGB - {self.current_image}", fontsize=14)
            
            # Imagen original
            axes[0].imshow(img)
            axes[0].set_title('Imagen original a color')
            axes[0].axis('off')
            
            # Histograma RGB
            axes[1].plot(hists['r'], color='red', alpha=0.7, label='Rojo')
            axes[1].plot(hists['g'], color='green', alpha=0.7, label='Verde')
            axes[1].plot(hists['b'], color='blue', alpha=0.7, label='Azul')
            axes[1].set_title('Histograma RGB')
            axes[1].set_xlabel('Nivel de intensidad')
            axes[1].set_ylabel('Frecuencia')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        else:  # Pestaña Resultado
            # Verificar qué tipo de resultado es
            if self.current_map and 'BINARIA' in self.current_map:
                # Si es binarizada, mostrar histograma simple
                hist = self.model.get_histogram_gray(self.current_image)
                img_result = self.model.get_result(self.current_image, self.current_map)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f"Histograma - Imagen Binarizada", fontsize=14)
                
                axes[0].imshow(img_result)
                axes[0].set_title('Imagen binarizada')
                axes[0].axis('off')
                
                axes[1].plot(hist, color='black')
                axes[1].fill_between(range(256), hist, alpha=0.3, color='gray')
                axes[1].set_title('Histograma de grises')
                axes[1].set_xlabel('Nivel de gris')
                axes[1].set_ylabel('Frecuencia')
                axes[1].grid(True, alpha=0.3)
                
            elif self.current_map:
                # Si es un mapa de color, mostrar histograma RGB del resultado
                # Convertir resultado a RGB para histograma
                img_result = self.model.get_result(self.current_image, self.current_map)
                img_result_bgr = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
                
                hist_r = cv2.calcHist([img_result_bgr], [2], None, [256], [0, 256])  # Rojo en BGR es canal 2
                hist_g = cv2.calcHist([img_result_bgr], [1], None, [256], [0, 256])  # Verde es canal 1
                hist_b = cv2.calcHist([img_result_bgr], [0], None, [256], [0, 256])  # Azul es canal 0
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f"Histograma RGB - {self.current_map}", fontsize=14)
                
                axes[0].imshow(img_result)
                axes[0].set_title(f'Imagen con mapa {self.current_map}')
                axes[0].axis('off')
                
                axes[1].plot(hist_r, color='red', alpha=0.7, label='Rojo')
                axes[1].plot(hist_g, color='green', alpha=0.7, label='Verde')
                axes[1].plot(hist_b, color='blue', alpha=0.7, label='Azul')
                axes[1].set_title('Histograma RGB')
                axes[1].set_xlabel('Nivel de intensidad')
                axes[1].set_ylabel('Frecuencia')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
            else:
                # Si no hay resultado, mostrar histograma de grises
                hist = self.model.get_histogram_gray(self.current_image)
                img_gray = self.model.get_gray_image(self.current_image)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f"Histograma - Escala de Grises", fontsize=14)
                
                axes[0].imshow(img_gray)
                axes[0].set_title('Imagen en grises')
                axes[0].axis('off')
                
                axes[1].plot(hist, color='black')
                axes[1].fill_between(range(256), hist, alpha=0.3, color='gray')
                axes[1].set_title('Histograma de grises')
                axes[1].set_xlabel('Nivel de gris')
                axes[1].set_ylabel('Frecuencia')
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        self.view.show_status("Histograma mostrado")
    def show_histogram(self):
        # Variante optimizada que obtiene histogramas según pestaña y tipo de mapa
        """Muestra el histograma según el tipo de imagen actual"""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        
        # Verificar qué pestaña está activa
        current_tab = self.view.tabs.currentIndex()
        
        if current_tab == 0:  # Pestaña Original
            # Mostrar histograma RGB de la imagen original a color
            hists = self.model.get_histogram_rgb(self.current_image)
            if hists is None:
                return
                
            img = self.model.get_image(self.current_image)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"Histograma RGB - {self.current_image}", fontsize=14)
            
            # Imagen original
            axes[0].imshow(img)
            axes[0].set_title('Imagen original a color')
            axes[0].axis('off')
            
            # Histograma RGB
            axes[1].plot(hists['r'], color='red', alpha=0.7, label='Rojo', linewidth=1.5)
            axes[1].plot(hists['g'], color='green', alpha=0.7, label='Verde', linewidth=1.5)
            axes[1].plot(hists['b'], color='blue', alpha=0.7, label='Azul', linewidth=1.5)
            axes[1].set_title('Histograma RGB')
            axes[1].set_xlabel('Nivel de intensidad (0-255)')
            axes[1].set_ylabel('Frecuencia')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        else:  # Pestaña Resultado
            if self.current_map:
                # Obtener histograma del resultado
                hists = self.model.get_histogram_result(self.current_image, self.current_map)
                img_result = self.model.get_result(self.current_image, self.current_map)
                
                if hists is None or img_result is None:
                    QMessageBox.warning(self.view, "Aviso", "No hay resultado para mostrar")
                    return
                
                # Determinar título según tipo de resultado
                if 'BINARIA' in self.current_map:
                    titulo = f"Histograma - {self.current_map}"
                    color_titulo = 'black'
                else:
                    titulo = f"Histograma RGB - {self.current_map}"
                    color_titulo = 'purple'
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(titulo, fontsize=14, color=color_titulo)
                
                # Imagen resultado
                axes[0].imshow(img_result)
                axes[0].set_title(f'Imagen con {self.current_map}')
                axes[0].axis('off')
                
                # Histograma
                if 'BINARIA' in self.current_map:
                    # Para binarizada, mostrar histograma simple (solo 0 y 255)
                    hist = hists['r']  # Usamos cualquier canal (son iguales)
                    axes[1].bar([0, 255], [hist[0], hist[255]], 
                            color=['black', 'white'], edgecolor='black', alpha=0.7)
                    axes[1].set_title('Histograma (0 = Negro, 255 = Blanco)')
                    axes[1].set_xlabel('Valor de píxel')
                    axes[1].set_ylabel('Frecuencia')
                else:
                    # Para mapas de color, mostrar RGB
                    axes[1].plot(hists['r'], color='red', alpha=0.7, label='Rojo', linewidth=1.5)
                    axes[1].plot(hists['g'], color='green', alpha=0.7, label='Verde', linewidth=1.5)
                    axes[1].plot(hists['b'], color='blue', alpha=0.7, label='Azul', linewidth=1.5)
                    axes[1].set_title('Histograma RGB')
                    axes[1].set_xlabel('Nivel de intensidad (0-255)')
                    axes[1].set_ylabel('Frecuencia')
                    axes[1].legend()
                
                axes[1].grid(True, alpha=0.3)
                
            else:
                # Si no hay mapa seleccionado, mostrar histograma de grises
                hist = self.model.get_histogram_gray(self.current_image)
                img_gray = self.model.get_gray_image(self.current_image)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle("Histograma - Escala de Grises", fontsize=14)
                
                axes[0].imshow(img_gray)
                axes[0].set_title('Imagen en grises')
                axes[0].axis('off')
                
                axes[1].plot(hist, color='black', linewidth=1.5)
                axes[1].fill_between(range(256), hist, alpha=0.3, color='gray')
                axes[1].set_title('Histograma de intensidades')
                axes[1].set_xlabel('Nivel de gris (0-255)')
                axes[1].set_ylabel('Frecuencia')
                axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        self.view.show_status("Histograma mostrado")
