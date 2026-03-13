"""Modelo del programa, maneja la carga y procesamiento de imágenes.
- Autor: Alejandra Michelle Mateo Garcia  & Leyva Triana Isis Valeria 
- Fecha: 20 de febrero del 2026
- Versión: 2.3 (bugfix)
- Descripción: Práctica 1 - "Explorando la Imagen Digital con Python"
               Correcciones:
               ✓ get_histogram_result ahora usa image_name correctamente
               ✓ get_histogram_gray valida image_name antes de procesar
               ✓ Validaciones defensivas en métodos de consulta
- Escuela: ESCOM-IPN
- Materia: Análisis de Imágenes
"""
import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class ImageModel:
    def __init__(self):
        # Inicializa los contenedores y define mapas de color disponibles
        self.images = {}       # {name: img_rgb}  — originales a color
        self.gray_images = {}  # {name: img_gray}  — versiones en gris
        # BUG FIX: la clave ahora es (image_name, map_name) para evitar
        # colisiones cuando se cargan varias imágenes en la misma sesión.
        # Se mantiene compatibilidad con las claves antiguas mediante
        # _result_key() que construye la clave compuesta.
        self.results = {}      # {(image_name, map_name): img_bgr}

        # Mapas de OpenCV
        self.colormaps = {
            "TWILIGHT": cv2.COLORMAP_TWILIGHT,
            "TURBO":    cv2.COLORMAP_TURBO,
            "VIRDIS":   cv2.COLORMAP_VIRIDIS,
            "PINK":     cv2.COLORMAP_PINK,
            "INFERNO":  cv2.COLORMAP_INFERNO,
            "WINTER":   cv2.COLORMAP_WINTER,
            "HSV":      cv2.COLORMAP_HSV,
            "PARULA":   cv2.COLORMAP_PARULA,
        }

        # Mapa personalizado (Morado a Fucsia)
        colores_personalizado = [
            (0.4, 0.0, 0.6),
            (0.6, 0.2, 0.8),
            (0.8, 0.3, 0.9),
            (1.0, 0.3, 0.6),
        ]
        self.mapa_personalizado = LinearSegmentedColormap.from_list(
            "Personalizado", colores_personalizado, N=256
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _result_key(self, image_name: str, map_name: str) -> tuple:
        """Construye la clave compuesta para self.results."""
        return (image_name, map_name)

    # ------------------------------------------------------------------
    # Carga y conversiones
    # ------------------------------------------------------------------

    def load_image(self, filepath: str) -> str:
        """Carga imagen a color desde disco y la guarda en formato RGB."""
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {filepath}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = filepath.split("/")[-1].split(".")[0]
        self.images[name] = img_rgb
        return name

    def convert_to_gray(self, image_name: str):
        """Convierte la imagen a color a escala de grises y la almacena."""
        if image_name not in self.images:
            return None
        img_rgb = self.images[image_name]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        self.gray_images[image_name] = img_gray
        return img_gray

    def get_rgb_components(self, image_name: str):
        """Retorna los canales R, G, B por separado de la imagen original."""
        if image_name not in self.images:
            return None, None, None
        r, g, b = cv2.split(self.images[image_name])
        return r, g, b

    # ------------------------------------------------------------------
    # Histogramas
    # ------------------------------------------------------------------

    def get_histogram_rgb(self, image_name: str):
        """Retorna histogramas para cada canal RGB de la imagen original."""
        if image_name not in self.images:
            return None
        img_rgb = self.images[image_name]
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
        return {
            "r": hist_r.flatten(),
            "g": hist_g.flatten(),
            "b": hist_b.flatten(),
        }

    def get_histogram_result(self, image_name: str, map_name: str):
        """Retorna histogramas RGB para el resultado de image_name+map_name.

        BUG FIX: la versión anterior ignoraba image_name y buscaba solo
        por map_name, devolviendo resultados de otra imagen si los mapas
        coincidían en nombre.
        """
        key = self._result_key(image_name, map_name)
        if key not in self.results:
            return None

        img_result_bgr = self.results[key]

        # Si es imagen en grises (1 canal), convertir a BGR para consistencia
        if len(img_result_bgr.shape) == 2:
            img_result_bgr = cv2.cvtColor(img_result_bgr, cv2.COLOR_GRAY2BGR)

        # OpenCV usa orden BGR internamente
        hist_b = cv2.calcHist([img_result_bgr], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_result_bgr], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img_result_bgr], [2], None, [256], [0, 256])

        return {
            "r": hist_r.flatten(),
            "g": hist_g.flatten(),
            "b": hist_b.flatten(),
        }

    def get_histogram_gray(self, image_name: str):
        """Retorna histograma de intensidades de la imagen en grises.

        BUG FIX: valida que image_name exista antes de convertir,
        evitando KeyError silenciosos.
        """
        if image_name not in self.images:
            return None
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        img_gray = self.gray_images[image_name]
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        return hist.flatten()

    # ------------------------------------------------------------------
    # Aplicación de mapas y binarización
    # ------------------------------------------------------------------

    def apply_colormap(self, image_name: str, map_name: str):
        """Aplica un mapa de color de OpenCV sobre la imagen en grises."""
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        img_gray = self.gray_images[image_name]
        result = cv2.applyColorMap(img_gray, self.colormaps[map_name])
        self.results[self._result_key(image_name, map_name)] = result
        return result

    def apply_personalized_map(self, image_name: str):
        """Aplica el mapa personalizado morado-fucsia."""
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        img_gray = self.gray_images[image_name]
        img_norm = img_gray / 255.0
        colored = self.mapa_personalizado(img_norm)
        colored_255 = (colored[:, :, :3] * 255).astype(np.uint8)
        result = cv2.cvtColor(colored_255, cv2.COLOR_RGB2BGR)
        self.results[self._result_key(image_name, "Personalizado")] = result
        return result

    def apply_threshold(self, image_name: str, threshold: int = 128, method: str = "fixed"):
        """Aplica binarización sobre la imagen en grises."""
        if image_name not in self.gray_images:
            self.convert_to_gray(image_name)
        img_gray = self.gray_images[image_name]

        if method == "fixed":
            _, binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        elif method == "otsu":
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # adaptive
            binary = cv2.adaptiveThreshold(
                img_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )

        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.results[self._result_key(image_name, f"BINARIA_{method}")] = binary_bgr
        return binary_bgr

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get_image(self, image_name: str):
        """Retorna la imagen original a color."""
        return self.images.get(image_name)

    def get_gray_image(self, image_name: str):
        """Retorna la imagen en escala de grises como RGB para la vista."""
        if image_name in self.gray_images:
            gray = self.gray_images[image_name]
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return None

    def get_result(self, image_name: str, map_name: str):
        """Devuelve el resultado procesado convertido a RGB para la vista."""
        key = self._result_key(image_name, map_name)
        if key in self.results:
            return cv2.cvtColor(self.results[key], cv2.COLOR_BGR2RGB)
        return None
