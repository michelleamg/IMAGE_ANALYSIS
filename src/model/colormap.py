"""Modelo del programa, maneja la carga y procesamiento de imágenes.
- Autor: Alejandra Michelle Mateo Garcia  & Leyva Triana Isis Valeria 
- Fecha: 20 de febrero del 2026
- Versión: 2.0
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
"""
import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class ImageModel:
    def __init__(self):
        # Inicializa los contenedores y define mapas de color disponibles
        self.images = {}  # diccionario para almacenar imágenes cargadas (originales a color)
        self.gray_images = {}  # diccionario para almacenar versiones en gris
        self.results = {}  # diccionario para almacenar resultados
        
        # Mapas de OpenCV
        self.colormaps = {
            "TWILIGHT": cv2.COLORMAP_TWILIGHT,
            "TURBO": cv2.COLORMAP_TURBO,
            "VIRDIS": cv2.COLORMAP_VIRIDIS,
            "PINK": cv2.COLORMAP_PINK,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "WINTER": cv2.COLORMAP_WINTER,
            "HSV": cv2.COLORMAP_HSV,
            "PARULA": cv2.COLORMAP_PARULA
        }
        
        # Mapa personalizado (Morado a Fucsia)
        colores_personalizado = [
            (0.4, 0.0, 0.6),   # Morado profundo
            (0.6, 0.2, 0.8),   # Morado medio
            (0.8, 0.3, 0.9),   # Morado claro
            (1.0, 0.3, 0.6),   # Fucsia
        ]
        self.mapa_personalizado = LinearSegmentedColormap.from_list("Personalizado", colores_personalizado, N=256)
    
    def load_image(self, filepath):
        # Lee la imagen desde disco y la guarda en formato RGB
        """Carga imagen a color"""
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {filepath}")
        
        # Convertir BGR a RGB para guardar
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        name = filepath.split('/')[-1].split('.')[0]
        self.images[name] = img_rgb  # Guardar original a color
        return name
    
    def convert_to_gray(self, image_name):
        # Genera y guarda la versión en escala de grises de la imagen
        """Convierte la imagen a color a escala de grises"""
        if image_name not in self.images:
            return None
        
        img_rgb = self.images[image_name]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        self.gray_images[image_name] = img_gray
        return img_gray
    
    def get_rgb_components(self, image_name):
        # Devuelve cada canal R, G y B por separado
        """Retorna los componentes R, G, B por separado de la imagen a color"""
        if image_name not in self.images:
            return None, None, None
        
        img_rgb = self.images[image_name]
        r, g, b = cv2.split(img_rgb)
        return r, g, b
    
    # ===== NUEVO: Método para histograma RGB =====
    def get_histogram_rgb(self, image_name):
        # Calcula histogramas individuales para los tres canales de la imagen original
        """Retorna histogramas para cada canal RGB de la imagen original"""
        if image_name not in self.images:
            return None
        
        img_rgb = self.images[image_name]  # Imagen a color en RGB
        
        # Calcular histograma para cada canal
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
        
        return {
            'r': hist_r.flatten(),
            'g': hist_g.flatten(),
            'b': hist_b.flatten()
        }
    
    # ===== NUEVO: Método para histograma de resultado =====
    def get_histogram_result(self, image_name, map_name):
        # Obtiene histogramas del resultado almacenado (mapa de color o binarización)
        """Retorna histogramas para el resultado aplicado"""
        if map_name not in self.results:
            return None
        
        img_result_bgr = self.results[map_name]  # Imagen en BGR
        
        # Si es imagen en grises (1 canal), convertir a BGR para consistencia
        if len(img_result_bgr.shape) == 2:
            img_result_bgr = cv2.cvtColor(img_result_bgr, cv2.COLOR_GRAY2BGR)
        
        # Calcular histogramas (OpenCV usa BGR)
        hist_b = cv2.calcHist([img_result_bgr], [0], None, [256], [0, 256])  # Azul
        hist_g = cv2.calcHist([img_result_bgr], [1], None, [256], [0, 256])  # Verde
        hist_r = cv2.calcHist([img_result_bgr], [2], None, [256], [0, 256])  # Rojo
        
        return {
            'r': hist_r.flatten(),
            'g': hist_g.flatten(),
            'b': hist_b.flatten()
        }
    
    def get_histogram_gray(self, image_name):
        # Calcula el histograma de intensidades en la versión en grises
        """Retorna histograma para imagen en grises"""
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        
        img_gray = self.gray_images[image_name]
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        return hist.flatten()
    
    def apply_colormap(self, image_name, map_name):
        # Aplica un mapa de color de OpenCV sobre la imagen en grises
        """Aplica mapa de color (requiere imagen en grises)"""
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        
        img_gray = self.gray_images[image_name]
        result = cv2.applyColorMap(img_gray, self.colormaps[map_name])
        self.results[map_name] = result
        return result
    
    def apply_personalized_map(self, image_name):
        # Genera mapa morado-fucsia personalizado y lo guarda
        """Aplica el mapa personalizado morado-fucsia"""
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        
        img_gray = self.gray_images[image_name]
        img_norm = img_gray / 255.0
        colored = self.mapa_personalizado(img_norm)
        colored_255 = (colored[:, :, :3] * 255).astype(np.uint8)
        result = cv2.cvtColor(colored_255, cv2.COLOR_RGB2BGR)
        self.results['Personalizado'] = result
        return result
    
    def apply_threshold(self, image_name, threshold=128, method='fixed'):
        # Ejecuta binarización fija, Otsu o adaptativa sobre la imagen en grises
        """Aplica binarización (requiere imagen en grises)"""
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        
        img_gray = self.gray_images[image_name]
        
        if method == 'fixed':
            _, binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        elif method == 'otsu':
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # adaptive
            binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.results[f'BINARIA_{method}'] = binary_bgr
        return binary_bgr
    
    def get_image(self, image_name):
        # Recupera la imagen original a color almacenada
        """Retorna la imagen original a color"""
        if image_name in self.images:
            return self.images[image_name]
        return None
    
    def get_gray_image(self, image_name):
        # Recupera la versión en grises en formato RGB para mostrarla
        """Retorna la imagen en escala de grises"""
        if image_name in self.gray_images:
            gray = self.gray_images[image_name]
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return None
    
    def get_result(self, image_name, map_name):
        # Devuelve el resultado procesado convertido a RGB para la vista
        if map_name in self.results:
            img = self.results[map_name]
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
