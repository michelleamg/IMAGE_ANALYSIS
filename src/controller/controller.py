"""Controlador del programa, maneja la lógica de interacción entre modelo y vista.
- Autor: Alejandra Michelle Mateo Garcia & Leyva Triana Isis Valeria
- Fecha: 20 de febrero del 2026
- Versión: 2.4 (bugfix)
- Descripción: Práctica 1 - "Explorando la Imagen Digital con Python"
               Correcciones:
               ✓ rgb_window y model_window guardados en self.* para evitar
                 que el garbage collector destruya las ventanas al instante
               ✓ Todas las llamadas a get_result y get_histogram_result
                 pasan image_name correctamente
               ✓ show_histogram dividido en métodos auxiliares privados
- Escuela: ESCOM-IPN
- Materia: Análisis de Imágenes
"""
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
import cv2
import numpy as np

from view.rgb_window import RGBComponentsWindow
from model.color_models import ColorModels
from view.color_models_window import ColorModelWindow
from view.histogram_window import HistogramWindow
from view.results_window import ResultsWindow
import model.practica3 as p3


class ImageController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_controller(self)
        self.current_image = None
        self.current_map = None
        # BUG FIX: referencias explícitas para que las ventanas no sean
        # destruidas por el garbage collector de Python.
        self.rgb_window = None
        self.model_window = None
        self.histogram_window = None
        self.results_window = None
        self.second_image = None   # imagen secundaria para operaciones de dos imágenes

    # ------------------------------------------------------------------
    # Acciones de carga
    # ------------------------------------------------------------------

    def load_image(self):
        """Abre diálogo de archivo y registra la imagen seleccionada."""
        filepath, _ = QFileDialog.getOpenFileName(
            self.view, "Seleccionar imagen", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if filepath:
            try:
                name = self.model.load_image(filepath)
                self.current_image = name
                img = self.model.get_image(name)
                self.view.set_image_name(name)
                self.view.show_original(img)
                self.view.show_status(f"Imagen cargada: {name}")
                # Actualizar histograma en vivo con canales RGB
                self._refresh_live_histogram()
            except Exception as e:
                QMessageBox.critical(self.view, "Error", str(e))

    # ------------------------------------------------------------------
    # Histograma en vivo
    # ------------------------------------------------------------------

    def _refresh_live_histogram(self):
        """Actualiza el histograma en vivo del sidebar según el contexto."""
        if not self.current_image:
            return
        if self.current_map and "BINARIA" in self.current_map:
            hist = self.model.get_histogram_gray(self.current_image)
            self.view.update_live_histogram(hist, mode="gray")
        elif self.current_map:
            hists = self.model.get_histogram_result(self.current_image, self.current_map)
            self.view.update_live_histogram(hists, mode="rgb")
        else:
            hists = self.model.get_histogram_rgb(self.current_image)
            self.view.update_live_histogram(hists, mode="rgb")

    # ------------------------------------------------------------------
    # Conversiones básicas
    # ------------------------------------------------------------------

    def convert_to_gray(self):
        """Convierte la imagen a color a escala de grises y la muestra."""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        self.model.convert_to_gray(self.current_image)
        img_gray = self.model.get_gray_image(self.current_image)
        if img_gray is not None:
            self.view.show_result(img_gray)
            self.view.show_status("Imagen convertida a escala de grises")
            hist = self.model.get_histogram_gray(self.current_image)
            self.view.update_live_histogram(hist, mode="gray")

    def show_rgb_components(self):
        """Muestra los componentes RGB en una ventana separada."""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        r, g, b = self.model.get_rgb_components(self.current_image)
        # BUG FIX: asignar a self.rgb_window para mantener la referencia viva
        self.rgb_window = RGBComponentsWindow(r, g, b, self.current_image)
        self.rgb_window.show()
        self.view.show_status("Componentes RGB mostrados")

    # ------------------------------------------------------------------
    # Mapas de color y binarización
    # ------------------------------------------------------------------

    def apply_map(self):
        """Aplica el mapa de color elegido (OpenCV o personalizado)."""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        map_name = self.view.map_combo.currentText()
        if "PERSONALIZADO" in map_name:
            self.model.apply_personalized_map(self.current_image)
            self.current_map = "Personalizado"
            map_display = "Personalizado (Morado-Fucsia)"
        else:
            self.model.apply_colormap(self.current_image, map_name)
            self.current_map = map_name
            map_display = map_name

        img_rgb = self.model.get_result(self.current_image, self.current_map)
        if img_rgb is not None:
            self.view.show_result(img_rgb)
            self.view.show_status(f"Mapa {map_display} aplicado")
            self._refresh_live_histogram()

    def apply_binary(self, method: str):
        """Ejecuta binarización con el método seleccionado."""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return
        threshold = self.view.threshold_slider.value()
        self.model.apply_threshold(self.current_image, threshold, method)
        self.current_map = f"BINARIA_{method}"
        img_rgb = self.model.get_result(self.current_image, self.current_map)
        if img_rgb is not None:
            self.view.show_result(img_rgb)
            self.view.show_status(f"Binarización {method} aplicada")
            self._refresh_live_histogram()

    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------

    def save_result(self):
        """Exporta a disco el resultado actualmente mostrado."""
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

    # ------------------------------------------------------------------
    # Estadísticas de histograma
    # ------------------------------------------------------------------

    def calculate_histogram_stats(self, hist_data):
        """Calcula estadísticas descriptivas a partir de datos de histograma."""
        if hist_data is None:
            return None

        if isinstance(hist_data, dict):
            if "r" in hist_data and "g" in hist_data and "b" in hist_data:
                hist = (hist_data["r"] + hist_data["g"] + hist_data["b"]) / 3
            else:
                return None
        else:
            hist = hist_data

        hist = hist.astype(np.float64)
        total_pixels = np.sum(hist)
        if total_pixels == 0:
            return None

        prob = hist / total_pixels
        niveles = np.arange(256)

        media = np.sum(niveles * prob)
        varianza = np.sum(((niveles - media) ** 2) * prob)
        desviacion = np.sqrt(varianza)
        asimetria = (
            np.sum(((niveles - media) ** 3) * prob) / (desviacion ** 3)
            if desviacion > 0 else 0
        )
        curtosis = (
            np.sum(((niveles - media) ** 4) * prob) / (desviacion ** 4) - 3
            if desviacion > 0 else 0
        )
        energia = np.sum(prob ** 2)
        entropia = -np.sum(prob * np.log2(prob + 1e-10))

        prob_acum = np.cumsum(prob)
        percentil_25 = int(np.searchsorted(prob_acum, 0.25))
        percentil_50 = int(np.searchsorted(prob_acum, 0.50))
        percentil_75 = int(np.searchsorted(prob_acum, 0.75))

        moda = int(np.argmax(hist))
        niveles_no_cero = niveles[hist > 0]
        rango_dinamico = (
            int(niveles_no_cero[-1] - niveles_no_cero[0])
            if len(niveles_no_cero) > 0 else 0
        )

        return {
            "media": media,
            "mediana": percentil_50,
            "varianza": varianza,
            "desviacion": desviacion,
            "asimetria": asimetria,
            "curtosis": curtosis,
            "energia": energia,
            "entropia": entropia,
            "moda": moda,
            "percentil_25": percentil_25,
            "percentil_75": percentil_75,
            "rango_dinamico": rango_dinamico,
            "min": int(niveles_no_cero[0]) if len(niveles_no_cero) > 0 else 0,
            "max": int(niveles_no_cero[-1]) if len(niveles_no_cero) > 0 else 0,
        }

    def format_stats_text(self, stats, channel_name: str = "") -> str:
        """Convierte las estadísticas en texto formateado para mostrar."""
        if stats is None:
            return "No hay datos suficientes"
        text = f"📊 Estadísticas {channel_name}:\n"
        text += "━" * 32 + "\n"
        text += f"Media:              {stats['media']:.2f}\n"
        text += f"Mediana:            {stats['mediana']}\n"
        text += f"Moda:               {stats['moda']}\n"
        text += f"Varianza:           {stats['varianza']:.2f}\n"
        text += f"Desv. estándar:     {stats['desviacion']:.2f}\n"
        text += f"Asimetría:          {stats['asimetria']:.3f}\n"
        text += f"Curtosis:           {stats['curtosis']:.3f}\n"
        text += f"Energía:            {stats['energia']:.6f}\n"
        text += f"Entropía:           {stats['entropia']:.3f} bits\n"
        text += f"Rango dinámico:     [{stats['min']}, {stats['max']}]\n"
        text += f"Percentil 25:       {stats['percentil_25']}\n"
        text += f"Percentil 75:       {stats['percentil_75']}\n"
        return text

    # ------------------------------------------------------------------
    # Histograma — métodos auxiliares privados
    # ------------------------------------------------------------------

    def _show_histogram_original(self):
        """Histograma RGB de la imagen original."""
        hists = self.model.get_histogram_rgb(self.current_image)
        if hists is None:
            return
        img = self.model.get_image(self.current_image)
        stats_data = {
            ch: self.calculate_histogram_stats(hists[ch])
            for ch in ("r", "g", "b")
        }
        self.histogram_window = HistogramWindow(
            mode="original",
            image_name=self.current_image,
            img=img,
            hists=hists,
            stats_data=stats_data,
        )
        self.histogram_window.show()

    def _show_histogram_gray(self):
        """Histograma de escala de grises."""
        hist = self.model.get_histogram_gray(self.current_image)
        img_gray = self.model.get_gray_image(self.current_image)
        if hist is None or img_gray is None:
            QMessageBox.warning(self.view, "Aviso", "No hay imagen en grises")
            return
        stats = self.calculate_histogram_stats(hist)
        self.histogram_window = HistogramWindow(
            mode="gray",
            image_name=self.current_image,
            img=img_gray,
            hists=hist,
            stats_data=stats,
        )
        self.histogram_window.show()

    def _show_histogram_binary(self, img_result, hist):
        """Histograma para imagen binarizada."""
        stats = self.calculate_histogram_stats(hist)
        self.histogram_window = HistogramWindow(
            mode="binary",
            image_name=f"{self.current_image} — {self.current_map}",
            img=img_result,
            hists=hist,
            stats_data=stats,
        )
        self.histogram_window.show()

    def _show_histogram_colormap(self, img_result, hists):
        """Histograma RGB para resultado con mapa de color."""
        stats_data = {
            ch: self.calculate_histogram_stats(hists[ch])
            for ch in ("r", "g", "b")
        }
        self.histogram_window = HistogramWindow(
            mode="colormap",
            image_name=f"{self.current_image} — {self.current_map}",
            img=img_result,
            hists=hists,
            stats_data=stats_data,
        )
        self.histogram_window.show()

    # ------------------------------------------------------------------
    # Histograma — punto de entrada público
    # ------------------------------------------------------------------

    def show_histogram(self):
        """Abre la ventana de histograma detallado según el contexto activo."""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return

        current_tab = self.view.tabs.currentIndex()

        if current_tab == 0:
            self._show_histogram_original()
        else:
            if self.current_map:
                hists = self.model.get_histogram_result(self.current_image, self.current_map)
                img_result = self.model.get_result(self.current_image, self.current_map)
                if hists is None or img_result is None:
                    QMessageBox.warning(self.view, "Aviso", "No hay resultado para mostrar")
                    return
                if "BINARIA" in self.current_map:
                    self._show_histogram_binary(img_result, hists["r"])
                else:
                    self._show_histogram_colormap(img_result, hists)
            else:
                self._show_histogram_gray()

        self.view.show_status("Histograma detallado abierto")

    # ------------------------------------------------------------------
    # Modelos de color
    # ------------------------------------------------------------------

    def apply_color_model(self):
        """Aplica transformación a diferentes modelos de color."""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return

        model_name = self.view.models_combo.currentText()
        img_rgb = self.model.get_image(self.current_image)
        if img_rgb is None:
            return

        model_data = None
        try:
            if "RGB" in model_name:
                r, g, b = cv2.split(img_rgb)
                model_data = {"r": r, "g": g, "b": b, "combined": img_rgb.copy()}
                display_name = "RGB"
            elif "CMYK" in model_name:
                model_data = ColorModels.rgb_to_cmyk(img_rgb)
                display_name = "CMYK"
            elif "HSV" in model_name:
                model_data = ColorModels.rgb_to_hsv(img_rgb)
                display_name = "HSV"
            elif "HSI" in model_name:
                model_data = ColorModels.rgb_to_hsi(img_rgb)
                display_name = "HSI"
            elif "YUV" in model_name:
                model_data = ColorModels.rgb_to_yuv(img_rgb)
                display_name = "YUV"
            elif "LAB" in model_name:
                model_data = ColorModels.rgb_to_lab(img_rgb)
                display_name = "LAB"
            elif "XYZ" in model_name:
                model_data = ColorModels.rgb_to_xyz(img_rgb)
                display_name = "XYZ"
            else:
                QMessageBox.warning(self.view, "Error", f"Modelo no reconocido: {model_name}")
                return

            if model_data:
                # BUG FIX: asignar a self.model_window para mantener referencia viva
                self.model_window = ColorModelWindow(model_data, display_name, img_rgb)
                self.model_window.show()
                self.view.show_status(f"Modelo {display_name} aplicado correctamente")

        except Exception as e:
            QMessageBox.critical(self.view, "Error", f"Error al aplicar modelo: {str(e)}")

    # ══════════════════════════════════════════════════════════════════════
    # PRÁCTICA 3-a — RUIDO
    # ══════════════════════════════════════════════════════════════════════

    def apply_noise(self):
        """Aplica ruido (sal y pimienta o gaussiano) y muestra comparativa."""
        if not self._require_image():
            return
        img = self.model.get_image(self.current_image)
        amount = self.view.noise_slider.value() / 100.0
        noise_type = self.view.noise_combo.currentText()

        if noise_type == "Sal y pimienta":
            noisy = p3.add_salt_pepper(img, amount=amount)
            badge = "ruido"
            subtitle = f"cantidad = {amount:.0%}"
        else:
            sigma = amount * 50           # slider 1–100 → sigma 0.5–50
            noisy = p3.add_gaussian_noise(img, sigma=sigma)
            badge = "ruido"
            subtitle = f"σ = {sigma:.1f}"

        self.view.show_result(noisy)
        self._store_result(noisy, f"RUIDO_{noise_type.upper().replace(' ', '_')}")

        cards = [
            {"title": "Original", "img": img, "badge": "ruido", "extra_info": "sin ruido"},
            {"title": f"Con {noise_type}", "img": noisy, "badge": badge,
             "extra_info": subtitle},
        ]
        h, w = img.shape[:2]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-a — {noise_type}",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen original",
            left_info=[
                ("Tipo de ruido", noise_type),
                ("Intensidad", f"{self.view.noise_slider.value()} %"),
                ("Dimensiones", f"{w} × {h}"),
            ],
            cols=2,
        )
        self.results_window.show()
        self.view.show_status(f"Ruido {noise_type} aplicado ({amount:.0%})")

    def apply_filter(self, filter_type: str):
        """Aplica filtro suavizador sobre el resultado actual."""
        if not self._require_image():
            return
        # Intentar tomar el resultado actual; si no hay, usar la imagen original
        img = self._get_current_display_img()

        if filter_type == "median":
            filtered = p3.apply_median_filter(img, ksize=3)
            name = "Filtro mediana (3×3)"
        else:
            filtered = p3.apply_gaussian_filter(img, ksize=5, sigma=1.0)
            name = "Filtro gaussiano (5×5, σ=1)"

        self.view.show_result(filtered)
        self._store_result(filtered, f"FILTRO_{filter_type.upper()}")

        cards = [
            {"title": "Antes del filtro", "img": img, "badge": "ruido"},
            {"title": name, "img": filtered, "badge": "ruido",
             "extra_info": "ruido reducido"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-a — {name}",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen de entrada",
            left_info=[("Filtro aplicado", name)],
            cols=2,
        )
        self.results_window.show()
        self.view.show_status(f"{name} aplicado")

    # ══════════════════════════════════════════════════════════════════════
    # PRÁCTICA 3-b — OPERACIONES ARITMÉTICAS, LÓGICAS Y RELACIONALES
    # ══════════════════════════════════════════════════════════════════════

    def apply_arithmetic(self, operation: str):
        """Suma, resta o multiplicación con el escalar del slider."""
        if not self._require_image():
            return
        img = self.model.get_image(self.current_image)
        scalar = self.view.scalar_slider.value()

        ops = {
            "add":      (p3.arith_add_scalar,      f"+ {scalar}",   "#6bffa0"),
            "subtract": (p3.arith_subtract_scalar,  f"− {scalar}",   "#ff6b6b"),
            "multiply": (p3.arith_multiply_scalar,   f"× {scalar/100:.2f}", "#ffee44"),
        }
        fn, label, _ = ops[operation]
        if operation == "multiply":
            result = fn(img, scalar / 100.0)
        else:
            result = fn(img, scalar)

        self.view.show_result(result)
        self._store_result(result, f"ARIT_{operation.upper()}")

        cards = [
            {"title": "Original", "img": img, "badge": "aritmética"},
            {"title": f"Imagen {label}", "img": result, "badge": "aritmética",
             "extra_info": f"escalar = {scalar}"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-b — Aritmética {label}",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen original",
            left_info=[
                ("Operación", label),
                ("Escalar", str(scalar)),
            ],
            cols=2,
        )
        self.results_window.show()
        self.view.show_status(f"Operación aritmética {label} aplicada")

    def apply_logic(self, operation: str):
        """AND, OR, XOR o NOT sobre la imagen actual (NOT es unaria; las demás usan segunda imagen)."""
        if not self._require_image():
            return
        img = self.model.get_image(self.current_image)

        if operation == "not":
            result = p3.logic_not(img)
            cards = [
                {"title": "Original", "img": img, "badge": "lógica"},
                {"title": "NOT (inversa)", "img": result, "badge": "lógica",
                 "extra_info": "inversión bit a bit"},
            ]
            info = [("Operación", "NOT (unaria)")]
            title_str = "NOT — inversión"
            col = 2
        else:
            # Necesitamos segunda imagen
            img2 = self._get_second_image()
            if img2 is None:
                return
            ops = {
                "and": (p3.logic_and, "AND — intersección"),
                "or":  (p3.logic_or,  "OR — unión"),
                "xor": (p3.logic_xor, "XOR — diferencia simétrica"),
            }
            fn, title_str = ops[operation]
            result = fn(img, img2)
            cards = [
                {"title": "Imagen A", "img": img, "badge": "lógica"},
                {"title": "Imagen B", "img": img2, "badge": "lógica"},
                {"title": title_str.split("—")[0].strip(), "img": result,
                 "badge": "lógica", "extra_info": title_str.split("—")[1].strip()},
            ]
            info = [
                ("Operación", operation.upper()),
                ("Imagen A", self.current_image),
                ("Imagen B", "segunda imagen"),
            ]
            col = 3

        self.view.show_result(result)
        self._store_result(result, f"LOGICA_{operation.upper()}")

        self.results_window = ResultsWindow(
            title=f"Práctica 3-b — {title_str}",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen A",
            left_info=info,
            cols=col,
        )
        self.results_window.show()
        self.view.show_status(f"Operación lógica {operation.upper()} aplicada")

    def apply_relational(self, operation: str):
        """Operaciones relacionales > < ≈ usando el umbral del slider de binarización."""
        if not self._require_image():
            return
        img = self.model.get_image(self.current_image)
        threshold = self.view.threshold_slider.value()

        ops = {
            "gt": (p3.relational_greater, f"> {threshold}  (mayor)"),
            "lt": (p3.relational_less,    f"< {threshold}  (menor)"),
            "eq": (p3.relational_equal,   f"≈ {threshold}  (igual ±10)"),
        }
        fn, label = ops[operation]
        result = fn(img, threshold)

        self.view.show_result(result)
        self._store_result(result, f"REL_{operation.upper()}")

        cards = [
            {"title": "Original", "img": img, "badge": "relacional"},
            {"title": f"Píxeles {label}", "img": result, "badge": "relacional",
             "extra_info": f"umbral = {threshold}"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-b — Relacional {label}",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen original",
            left_info=[
                ("Operación", label),
                ("Umbral", str(threshold)),
            ],
            cols=2,
        )
        self.results_window.show()
        self.view.show_status(f"Operación relacional {label} aplicada")

    # ══════════════════════════════════════════════════════════════════════
    # PRÁCTICA 3-c — COMPONENTES CONEXAS
    # ══════════════════════════════════════════════════════════════════════

    def apply_connected_components(self, connectivity: int):
        """Etiqueta y cuenta objetos con vecindad 4 u 8."""
        if not self._require_image():
            return
        img = self._get_current_display_img()
        result = p3.connected_components(img, connectivity=connectivity)

        n = result["num_objects"]
        self.view.show_result(result["contours_img"])
        self._store_result(result["contours_img"], f"CC_V{connectivity}")

        # Construir info de estadísticas de objetos
        stats_lines = [(f"Objeto {s['id']}", f"área={s['area']} cx={s['cx']} cy={s['cy']}")
                       for s in result["stats"][:8]]  # máx 8 para no saturar el panel

        cards = [
            {"title": f"Vecindad-{connectivity}  (etiquetado)",
             "img": result["labels_img"],
             "badge": "etiquetado",
             "extra_info": f"{n} objeto(s) detectado(s)"},
            {"title": f"Vecindad-{connectivity}  (contornos)",
             "img": result["contours_img"],
             "badge": "etiquetado",
             "extra_info": f"contornos verdes numerados"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-c — Vecindad-{connectivity}",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen binaria",
            left_info=[
                ("Vecindad", str(connectivity)),
                ("Objetos detectados", str(n)),
            ] + stats_lines,
            cols=2,
        )
        self.results_window.show()
        self.view.show_status(f"Vecindad-{connectivity}: {n} objeto(s) detectado(s)")

    def compare_connected_components(self):
        """Compara vecindad-4 vs vecindad-8 en una sola ventana."""
        if not self._require_image():
            return
        img = self._get_current_display_img()
        r4, r8 = p3.compare_connectivity(img)

        n4, n8 = r4["num_objects"], r8["num_objects"]
        dif = abs(n4 - n8)

        cards = [
            {"title": "Vecindad-4  (etiquetado)",
             "img": r4["labels_img"], "badge": "etiquetado",
             "extra_info": f"{n4} objeto(s)"},
            {"title": "Vecindad-4  (contornos)",
             "img": r4["contours_img"], "badge": "etiquetado",
             "extra_info": "conectividad ortogonal"},
            {"title": "Vecindad-8  (etiquetado)",
             "img": r8["labels_img"], "badge": "etiquetado",
             "extra_info": f"{n8} objeto(s)"},
            {"title": "Vecindad-8  (contornos)",
             "img": r8["contours_img"], "badge": "etiquetado",
             "extra_info": "incluye diagonales"},
        ]
        self.results_window = ResultsWindow(
            title="Práctica 3-c — Comparación V4 vs V8",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen binaria",
            left_info=[
                ("Objetos V-4", str(n4)),
                ("Objetos V-8", str(n8)),
                ("Diferencia", str(dif)),
                ("", "V-8 detecta más" if n8 > n4
                     else "V-4 detecta más" if n4 > n8
                     else "misma detección"),
            ],
            cols=2,
        )
        self.results_window.show()
        self.view.show_status(
            f"Comparación: V-4={n4} obj  V-8={n8} obj  diferencia={dif}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Helpers internos
    # ══════════════════════════════════════════════════════════════════════

    def _require_image(self) -> bool:
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return False
        return True

    def _store_result(self, img_rgb: np.ndarray, key: str):
        """Guarda un resultado en el modelo con clave compuesta."""
        import cv2 as _cv2
        bgr = _cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR) if img_rgb.ndim == 3 else img_rgb
        self.model.results[self.model._result_key(self.current_image, key)] = bgr
        self.current_map = key
        self._refresh_live_histogram()

    def _get_current_display_img(self) -> np.ndarray:
        """Retorna la imagen actualmente visible (original o resultado)."""
        if self.current_map:
            img = self.model.get_result(self.current_image, self.current_map)
            if img is not None:
                return img
        return self.model.get_image(self.current_image)

    def _get_binary_img(self) -> np.ndarray:
        """Retorna la imagen binaria activa.

        Prioridad:
          1. Si current_map ya apunta a una binarización (BINARIA_*) → la usa.
          2. Si no → binariza la imagen original con Otsu y lo registra.

        Siempre actualiza la pestaña Resultado para que el usuario vea la
        imagen binaria que se usará como base antes de operar.
        """
        if self.current_map and "BINARIA" in self.current_map:
            img = self.model.get_result(self.current_image, self.current_map)
            if img is not None:
                return img

        # Auto-binarizar con Otsu y mostrar resultado
        threshold = self.view.threshold_slider.value()
        self.model.apply_threshold(self.current_image, threshold, "otsu")
        self.current_map = "BINARIA_otsu"
        img = self.model.get_result(self.current_image, self.current_map)
        self.view.show_result(img)
        self.view.show_status(
            "Binarización Otsu aplicada automáticamente antes de la operación"
        )
        self._refresh_live_histogram()
        return img

    def _get_second_image_binary(self) -> np.ndarray | None:
        """Pide al usuario una segunda imagen y la binariza con Otsu."""
        from PyQt5.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getOpenFileName(
            self.view,
            "Seleccionar segunda imagen (Imagen B) — se binarizará con Otsu",
            "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp)",
        )
        if not filepath:
            return None
        import cv2 as _cv2
        img = _cv2.imread(filepath)
        if img is None:
            QMessageBox.critical(self.view, "Error", "No se pudo cargar la segunda imagen")
            return None
        rgb  = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
        gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
        _, binary = _cv2.threshold(gray, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        return _cv2.cvtColor(binary, _cv2.COLOR_GRAY2RGB)

    # ══════════════════════════════════════════════════════════════════════
    # PRÁCTICA 3-a — RUIDO  (sobre imagen binaria)
    # ══════════════════════════════════════════════════════════════════════

    def apply_noise(self):
        """Aplica ruido sobre la imagen binaria y muestra secuencia completa."""
        if not self._require_image():
            return

        original   = self.model.get_image(self.current_image)
        binary     = self._get_binary_img()
        amount     = self.view.noise_slider.value() / 100.0
        noise_type = self.view.noise_combo.currentText()

        if noise_type == "Sal y pimienta":
            noisy    = p3.add_salt_pepper(binary, amount=amount)
            subtitle = f"cantidad = {amount:.0%}"
        else:
            sigma    = amount * 50
            noisy    = p3.add_gaussian_noise(binary, sigma=sigma)
            subtitle = f"σ = {sigma:.1f}"

        self.view.show_result(noisy)
        self._store_result(noisy, f"RUIDO_{noise_type.upper().replace(' ', '_')}")
        h, w = original.shape[:2]

        cards = [
            {"title": "Original (color)",        "img": original, "badge": "ruido",
             "extra_info": "imagen cargada"},
            {"title": "Binaria (Otsu)",           "img": binary,   "badge": "ruido",
             "extra_info": "base para el ruido"},
            {"title": f"Binaria + {noise_type}",  "img": noisy,    "badge": "ruido",
             "extra_info": subtitle},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-a — {noise_type} sobre binaria",
            ref_img=binary,
            cards_data=cards,
            left_title="Imagen binaria",
            left_info=[
                ("Tipo de ruido", noise_type),
                ("Intensidad",    f"{self.view.noise_slider.value()} %"),
                ("Dimensiones",   f"{w} × {h}"),
                ("Base",          "imagen binaria (Otsu)"),
            ],
            cols=3,
        )
        self.results_window.show()
        self.view.show_status(
            f"Ruido {noise_type} aplicado sobre imagen binaria ({amount:.0%})"
        )

    def apply_filter(self, filter_type: str):
        """Filtro suavizador + re-binarización. Muestra secuencia de 3 pasos."""
        if not self._require_image():
            return
        img = self._get_current_display_img()

        if filter_type == "median":
            filtered = p3.apply_median_filter(img, ksize=3)
            name = "Filtro mediana (3×3)"
        else:
            filtered = p3.apply_gaussian_filter(img, ksize=5, sigma=1.0)
            name = "Filtro gaussiano (5×5, σ=1)"

        # Re-binarizar para devolver una imagen binaria limpia
        import cv2 as _cv2
        gray_f = _cv2.cvtColor(filtered, _cv2.COLOR_RGB2GRAY) if filtered.ndim == 3 else filtered
        _, clean_bin = _cv2.threshold(gray_f, 127, 255, _cv2.THRESH_BINARY)
        clean_rgb = _cv2.cvtColor(clean_bin, _cv2.COLOR_GRAY2RGB)

        self.view.show_result(clean_rgb)
        self._store_result(clean_rgb, f"FILTRO_{filter_type.upper()}")

        cards = [
            {"title": "Con ruido (binaria)",     "img": img,       "badge": "ruido"},
            {"title": name,                      "img": filtered,   "badge": "ruido",
             "extra_info": "suavizado"},
            {"title": "Re-binarizada (limpia)",  "img": clean_rgb,  "badge": "ruido",
             "extra_info": "lista para etiquetado"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-a — {name}",
            ref_img=img,
            cards_data=cards,
            left_title="Imagen con ruido",
            left_info=[
                ("Filtro aplicado",  name),
                ("Re-binarización", "umbral 127"),
            ],
            cols=3,
        )
        self.results_window.show()
        self.view.show_status(f"{name} aplicado — imagen re-binarizada")

    # ══════════════════════════════════════════════════════════════════════
    # PRÁCTICA 3-b — OPERACIONES ARITMÉTICAS, LÓGICAS Y RELACIONALES
    #                (todas sobre imagen binaria)
    # ══════════════════════════════════════════════════════════════════════

    def apply_arithmetic(self, operation: str):
        """Suma, resta o multiplicación escalar sobre la imagen binaria."""
        if not self._require_image():
            return

        original = self.model.get_image(self.current_image)
        binary   = self._get_binary_img()
        scalar   = self.view.scalar_slider.value()

        ops = {
            "add":      (p3.arith_add_scalar,     f"+ {scalar}"),
            "subtract": (p3.arith_subtract_scalar, f"− {scalar}"),
            "multiply": (p3.arith_multiply_scalar,  f"× {scalar/100:.2f}"),
        }
        fn, label = ops[operation]
        result = fn(binary, scalar / 100.0 if operation == "multiply" else scalar)

        self.view.show_result(result)
        self._store_result(result, f"ARIT_{operation.upper()}")

        cards = [
            {"title": "Original (color)", "img": original, "badge": "aritmética"},
            {"title": "Binaria (base)",   "img": binary,   "badge": "aritmética",
             "extra_info": "imagen binarizada"},
            {"title": f"Binaria {label}", "img": result,   "badge": "aritmética",
             "extra_info": f"escalar = {scalar}"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-b — Aritmética {label} (binaria)",
            ref_img=binary,
            cards_data=cards,
            left_title="Imagen binaria",
            left_info=[
                ("Operación", label),
                ("Escalar",   str(scalar)),
                ("Base",      "imagen binaria"),
            ],
            cols=3,
        )
        self.results_window.show()
        self.view.show_status(f"Operación aritmética {label} sobre imagen binaria")

    def apply_logic(self, operation: str):
        """AND, OR, XOR, NOT sobre imágenes binarias."""
        if not self._require_image():
            return

        original = self.model.get_image(self.current_image)
        binary_a = self._get_binary_img()

        if operation == "not":
            result = p3.logic_not(binary_a)
            cards = [
                {"title": "Original (color)", "img": original, "badge": "lógica"},
                {"title": "Binaria A",        "img": binary_a, "badge": "lógica"},
                {"title": "NOT (inversa)",    "img": result,   "badge": "lógica",
                 "extra_info": "inversión bit a bit"},
            ]
            info      = [("Operación", "NOT (unaria)"), ("Base", "imagen binaria")]
            title_str = "NOT — inversión binaria"
            col = 3
        else:
            binary_b = self._get_second_image_binary()
            if binary_b is None:
                return
            ops_map = {
                "and": (p3.logic_and, "AND — intersección"),
                "or":  (p3.logic_or,  "OR  — unión"),
                "xor": (p3.logic_xor, "XOR — diferencia simétrica"),
            }
            fn, title_str = ops_map[operation]
            result = fn(binary_a, binary_b)
            cards = [
                {"title": "Original A (color)", "img": original,  "badge": "lógica"},
                {"title": "Binaria A",           "img": binary_a,  "badge": "lógica"},
                {"title": "Binaria B",           "img": binary_b,  "badge": "lógica"},
                {"title": title_str.split("—")[0].strip(), "img": result,
                 "badge": "lógica",
                 "extra_info": title_str.split("—")[1].strip()},
            ]
            info = [
                ("Operación", operation.upper()),
                ("Imagen A",  self.current_image),
                ("Imagen B",  "segunda (binarizada Otsu)"),
                ("Base",      "ambas imágenes binarias"),
            ]
            col = 4

        self.view.show_result(result)
        self._store_result(result, f"LOGICA_{operation.upper()}")

        self.results_window = ResultsWindow(
            title=f"Práctica 3-b — {title_str}",
            ref_img=binary_a,
            cards_data=cards,
            left_title="Imagen binaria A",
            left_info=info,
            cols=col,
        )
        self.results_window.show()
        self.view.show_status(
            f"Operación lógica {operation.upper()} sobre imágenes binarias"
        )

    def apply_relational(self, operation: str):
        """Operaciones relacionales > < ≈ sobre la imagen binaria."""
        if not self._require_image():
            return

        original  = self.model.get_image(self.current_image)
        binary    = self._get_binary_img()
        threshold = self.view.threshold_slider.value()

        ops = {
            "gt": (p3.relational_greater, f"> {threshold}  (mayor)"),
            "lt": (p3.relational_less,    f"< {threshold}  (menor)"),
            "eq": (p3.relational_equal,   f"≈ {threshold}  (igual ±10)"),
        }
        fn, label = ops[operation]
        result = fn(binary, threshold)

        self.view.show_result(result)
        self._store_result(result, f"REL_{operation.upper()}")

        cards = [
            {"title": "Original (color)",    "img": original, "badge": "relacional"},
            {"title": "Binaria (base)",       "img": binary,   "badge": "relacional",
             "extra_info": "imagen binarizada"},
            {"title": f"Relacional {label}",  "img": result,   "badge": "relacional",
             "extra_info": f"umbral = {threshold}"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-b — Relacional {label} (binaria)",
            ref_img=binary,
            cards_data=cards,
            left_title="Imagen binaria",
            left_info=[
                ("Operación", label),
                ("Umbral",    str(threshold)),
                ("Base",      "imagen binaria"),
            ],
            cols=3,
        )
        self.results_window.show()
        self.view.show_status(f"Relacional {label} sobre imagen binaria")

    # ══════════════════════════════════════════════════════════════════════
    # PRÁCTICA 3-c — COMPONENTES CONEXAS (sobre imagen binaria)
    # ══════════════════════════════════════════════════════════════════════

    def apply_connected_components(self, connectivity: int):
        """Etiqueta y cuenta objetos con vecindad 4 u 8 sobre imagen binaria."""
        if not self._require_image():
            return

        original = self.model.get_image(self.current_image)
        binary   = self._get_binary_img()
        result   = p3.connected_components(binary, connectivity=connectivity)

        n = result["num_objects"]
        self.view.show_result(result["contours_img"])
        self._store_result(result["contours_img"], f"CC_V{connectivity}")

        stats_lines = [
            (f"Objeto {s['id']}", f"área={s['area']}  cx={s['cx']}  cy={s['cy']}")
            for s in result["stats"][:8]
        ]
        cards = [
            {"title": "Original (color)",            "img": original,              "badge": "etiquetado"},
            {"title": "Binaria (base)",               "img": binary,                "badge": "etiquetado",
             "extra_info": "imagen binarizada"},
            {"title": f"V-{connectivity} etiquetado", "img": result["labels_img"],  "badge": "etiquetado",
             "extra_info": f"{n} objeto(s)"},
            {"title": f"V-{connectivity} contornos",  "img": result["contours_img"],"badge": "etiquetado",
             "extra_info": "contornos numerados"},
        ]
        self.results_window = ResultsWindow(
            title=f"Práctica 3-c — Vecindad-{connectivity} (binaria)",
            ref_img=binary,
            cards_data=cards,
            left_title="Imagen binaria",
            left_info=[
                ("Vecindad",           str(connectivity)),
                ("Objetos detectados", str(n)),
            ] + stats_lines,
            cols=4,
        )
        self.results_window.show()
        self.view.show_status(
            f"Vecindad-{connectivity}: {n} objeto(s) en imagen binaria"
        )

    def compare_connected_components(self):
        """Compara V-4 vs V-8 sobre la imagen binaria activa."""
        if not self._require_image():
            return

        original = self.model.get_image(self.current_image)
        binary   = self._get_binary_img()
        r4, r8   = p3.compare_connectivity(binary)

        n4, n8 = r4["num_objects"], r8["num_objects"]
        dif    = abs(n4 - n8)

        cards = [
            {"title": "Original (color)",  "img": original,          "badge": "etiquetado"},
            {"title": "Binaria (base)",    "img": binary,            "badge": "etiquetado",
             "extra_info": "imagen binarizada"},
            {"title": "V-4  etiquetado",  "img": r4["labels_img"],  "badge": "etiquetado",
             "extra_info": f"{n4} objeto(s)"},
            {"title": "V-4  contornos",   "img": r4["contours_img"],"badge": "etiquetado",
             "extra_info": "conectividad ortogonal"},
            {"title": "V-8  etiquetado",  "img": r8["labels_img"],  "badge": "etiquetado",
             "extra_info": f"{n8} objeto(s)"},
            {"title": "V-8  contornos",   "img": r8["contours_img"],"badge": "etiquetado",
             "extra_info": "incluye diagonales"},
        ]
        self.results_window = ResultsWindow(
            title="Práctica 3-c — Comparación V4 vs V8 (binaria)",
            ref_img=binary,
            cards_data=cards,
            left_title="Imagen binaria",
            left_info=[
                ("Objetos V-4", str(n4)),
                ("Objetos V-8", str(n8)),
                ("Diferencia",  str(dif)),
                ("Resultado",   "V-8 detecta más" if n8 > n4
                                else "V-4 detecta más" if n4 > n8
                                else "misma detección"),
            ],
            cols=3,
        )
        self.results_window.show()
        self.view.show_status(
            f"Comparación binaria: V-4={n4} obj  V-8={n8} obj  diferencia={dif}"
        )

