import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class ImageModel:
    def __init__(self):
        self.images = {}
        self.results = {}
        
        # Mapas de OpenCV
        self.colormaps = {
            "JET": cv2.COLORMAP_JET,
            "HOT": cv2.COLORMAP_HOT,
            "COOL": cv2.COLORMAP_COOL,
            "SPRING": cv2.COLORMAP_SPRING,
            "SUMMER": cv2.COLORMAP_SUMMER,
            "WINTER": cv2.COLORMAP_WINTER,
            "HSV": cv2.COLORMAP_HSV,
            "PARULA": cv2.COLORMAP_PARULA
        }
        
        # Mapa personalizado (MORADOS A FUCSIAS)
        colores_personalizado = [
            (0.4, 0.0, 0.6),   # Morado profundo
            (0.6, 0.2, 0.8),   # Morado medio
            (0.8, 0.3, 0.9),   # Morado claro
            (1.0, 0.3, 0.6),   # Fucsia
        ]
        self.mapa_personalizado = LinearSegmentedColormap.from_list("Personalizado", colores_personalizado, N=256)
    
    def load_image(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_color = cv2.imread(filepath)
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        name = filepath.split('/')[-1].split('.')[0]
        self.images[name] = img
        return name
    
    def apply_colormap(self, image_name, map_name):
        img = self.images[image_name]
        result = cv2.applyColorMap(img, self.colormaps[map_name])
        self.results[map_name] = result
        return result
    
    def apply_personalized_map(self, image_name):  # <-- ESTE MÉTODO ESTABA FALTANDO
        """Aplica el mapa personalizado (morados a fucsias)"""
        img = self.images[image_name]
        img_norm = img / 255.0
        colored = self.mapa_personalizado(img_norm)
        colored_255 = (colored[:, :, :3] * 255).astype(np.uint8)
        result = cv2.cvtColor(colored_255, cv2.COLOR_RGB2BGR)
        self.results['Personalizado'] = result
        return result
    
    def get_image(self, image_name):
        img = self.images[image_name]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    def get_result(self, image_name, map_name):
        if map_name in self.results:
            img = self.results[map_name]
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None