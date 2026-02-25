import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class ImageModel:
    def __init__(self):
        self.images = {}
        self.results = {}
        self.current_image = None
        
        # Mapas de color de OpenCV
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
        
        # Mapa pastel
        self.pastel_map = self.create_pastel_map()
    
    def create_pastel_map(self):
        from matplotlib.colors import LinearSegmentedColormap
        """colores_pastel = [
            (1.0, 0.8, 0.9),  # rosa
            (0.8, 1.0, 0.8),  # menta
            (0.8, 0.9, 1.0),  # lavanda
            (1.0, 1.0, 0.8),  # amarillo
            (0.9, 0.8, 1.0)   # violeta
        ]"""
        colores_morado_fucsia = [
            (0.4, 0.0, 0.6),   # Morado profundo
            (0.6, 0.2, 0.8),   # Morado medio
            (0.8, 0.3, 0.9),   # Morado claro / Lavanda
            (0.9, 0.4, 0.8),   # Rosa morado
            (1.0, 0.5, 0.7),   # Rosa intenso
            (1.0, 0.3, 0.6),   # Fucsia
            (1.0, 0.0, 0.5)    # Fucsia brillante
]
        return LinearSegmentedColormap.from_list("Morado Fucsia", colores_morado_fucsia, N=256)
    
    def load_image(self, filepath):
        """Carga imagen en grises"""
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img_color = cv2.imread(filepath)
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        name = filepath.split('/')[-1].split('.')[0]
        self.images[name] = img
        self.current_image = name
        return name
    
    def apply_colormap(self, image_name, map_name):
        """Aplica mapa de OpenCV"""
        img = self.images[image_name]
        result = cv2.applyColorMap(img, self.colormaps[map_name])
        
        if image_name not in self.results:
            self.results[image_name] = {}
        self.results[image_name][map_name] = result
        return result
    
    def apply_pastel_map(self, image_name):
        """Aplica mapa pastel"""
        img = self.images[image_name]
        
        # Normalizar y aplicar mapa pastel
        img_norm = img / 255.0
        colored = self.pastel_map(img_norm)
        
        # Convertir a formato OpenCV
        colored_255 = (colored[:, :, :3] * 255).astype(np.uint8)
        colored_bgr = cv2.cvtColor(colored_255, cv2.COLOR_RGB2BGR)
        
        if image_name not in self.results:
            self.results[image_name] = {}
        self.results[image_name]['PASTEL'] = colored_bgr
        return colored_bgr
    
    def get_image(self, image_name):
        """Retorna imagen para mostrar"""
        img = self.images[image_name]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    def get_result(self, image_name, map_name):
        """Retorna resultado para mostrar"""
        if image_name in self.results and map_name in self.results[image_name]:
            img = self.results[image_name][map_name]
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None