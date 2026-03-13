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
import matplotlib.pyplot as plt
import cv2
import numpy as np

from view.rgb_window import RGBComponentsWindow
from model.color_models import ColorModels
from view.color_models_window import ColorModelWindow


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
        """Muestra histograma RGB de la imagen original con estadísticas."""
        hists = self.model.get_histogram_rgb(self.current_image)
        if hists is None:
            return
        img = self.model.get_image(self.current_image)

        stats_r = self.calculate_histogram_stats(hists["r"])
        stats_g = self.calculate_histogram_stats(hists["g"])
        stats_b = self.calculate_histogram_stats(hists["b"])

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Análisis de Histograma — {self.current_image}",
            fontsize=14, fontweight="bold",
        )
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

        ax_img = fig.add_subplot(gs[0, :2])
        ax_img.imshow(img)
        ax_img.set_title("Imagen original a color")
        ax_img.axis("off")

        ax_hist = fig.add_subplot(gs[0, 2])
        ax_hist.plot(hists["r"], color="red",   alpha=0.7, label="Rojo",  linewidth=1.5)
        ax_hist.plot(hists["g"], color="green", alpha=0.7, label="Verde", linewidth=1.5)
        ax_hist.plot(hists["b"], color="blue",  alpha=0.7, label="Azul",  linewidth=1.5)
        ax_hist.set_title("Histograma RGB")
        ax_hist.set_xlabel("Nivel de intensidad")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)

        for ax, stats, label, color in [
            (fig.add_subplot(gs[1, 0]), stats_r, "Canal Rojo",  "#ffeeee"),
            (fig.add_subplot(gs[1, 1]), stats_g, "Canal Verde", "#eeffee"),
            (fig.add_subplot(gs[1, 2]), stats_b, "Canal Azul",  "#eeeeff"),
        ]:
            ax.axis("off")
            ax.text(
                0.1, 0.95, self.format_stats_text(stats, label),
                transform=ax.transAxes, fontsize=9, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

    def _show_histogram_gray(self):
        """Muestra histograma de intensidades en escala de grises."""
        hist = self.model.get_histogram_gray(self.current_image)
        img_gray = self.model.get_gray_image(self.current_image)
        if hist is None or img_gray is None:
            QMessageBox.warning(self.view, "Aviso", "No hay imagen en grises")
            return

        stats = self.calculate_histogram_stats(hist)
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("Análisis de Histograma — Escala de Grises",
                     fontsize=14, fontweight="bold")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_gray, cmap="gray")
        ax_img.set_title("Imagen en escala de grises")
        ax_img.axis("off")

        ax_hist = fig.add_subplot(gs[0, 1])
        ax_hist.plot(hist, color="black", linewidth=1.5)
        ax_hist.fill_between(range(256), hist, alpha=0.3, color="gray")
        ax_hist.set_title("Histograma de intensidades")
        ax_hist.set_xlabel("Nivel de gris")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.grid(True, alpha=0.3)

        if stats:
            ax_hist.axvline(x=stats["media"],   color="red",   linestyle="--",
                            linewidth=2, label=f"Media: {stats['media']:.1f}")
            ax_hist.axvline(x=stats["mediana"], color="green", linestyle=":",
                            linewidth=2, label=f"Mediana: {stats['mediana']}")
            ax_hist.axvline(x=stats["moda"],    color="blue",  linestyle="-.",
                            linewidth=2, label=f"Moda: {stats['moda']}")
            ax_hist.legend(fontsize=8)

        ax_stats = fig.add_subplot(gs[1, :])
        ax_stats.axis("off")
        ax_stats.text(
            0.5, 0.5, self.format_stats_text(stats, "Intensidad"),
            transform=ax_stats.transAxes, fontsize=10, fontfamily="monospace",
            horizontalalignment="center", verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9),
        )

    def _show_histogram_binary(self, img_result, hist):
        """Muestra histograma simplificado para imágenes binarizadas."""
        stats = self.calculate_histogram_stats(hist)
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(
            f"Análisis de Histograma — {self.current_map}",
            fontsize=14, fontweight="bold",
        )
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img_result)
        ax_img.set_title(f"Imagen binarizada ({self.current_map.split('_')[1]})")
        ax_img.axis("off")

        ax_hist = fig.add_subplot(gs[0, 1])
        ax_hist.bar(
            [0, 255], [hist[0], hist[255]],
            color=["black", "white"], edgecolor="black", alpha=0.7, width=50,
        )
        ax_hist.set_title("Histograma (0 = Negro, 255 = Blanco)")
        ax_hist.set_xlabel("Valor de píxel")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.grid(True, alpha=0.3)

        ax_stats = fig.add_subplot(gs[1, :])
        ax_stats.axis("off")
        ax_stats.text(
            0.5, 0.5, self.format_stats_text(stats, ""),
            transform=ax_stats.transAxes, fontsize=10, fontfamily="monospace",
            horizontalalignment="center", verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9),
        )

    def _show_histogram_colormap(self, img_result, hists):
        """Muestra histograma RGB para resultados con mapa de color."""
        stats_r = self.calculate_histogram_stats(hists["r"])
        stats_g = self.calculate_histogram_stats(hists["g"])
        stats_b = self.calculate_histogram_stats(hists["b"])

        fig = plt.figure(figsize=(14, 8))
        fig.suptitle(
            f"Análisis de Histograma RGB — {self.current_map}",
            fontsize=14, fontweight="bold",
        )
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

        ax_img = fig.add_subplot(gs[0, :2])
        ax_img.imshow(img_result)
        ax_img.set_title(f"Imagen con mapa {self.current_map}")
        ax_img.axis("off")

        ax_hist = fig.add_subplot(gs[0, 2])
        ax_hist.plot(hists["r"], color="red",   alpha=0.7, label="Rojo",  linewidth=1.5)
        ax_hist.plot(hists["g"], color="green", alpha=0.7, label="Verde", linewidth=1.5)
        ax_hist.plot(hists["b"], color="blue",  alpha=0.7, label="Azul",  linewidth=1.5)
        ax_hist.set_title("Histograma RGB")
        ax_hist.set_xlabel("Nivel de intensidad")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)

        for ax, stats, label, color in [
            (fig.add_subplot(gs[1, 0]), stats_r, "Canal Rojo",  "#ffeeee"),
            (fig.add_subplot(gs[1, 1]), stats_g, "Canal Verde", "#eeffee"),
            (fig.add_subplot(gs[1, 2]), stats_b, "Canal Azul",  "#eeeeff"),
        ]:
            ax.axis("off")
            ax.text(
                0.1, 0.95, self.format_stats_text(stats, label),
                transform=ax.transAxes, fontsize=9, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

    # ------------------------------------------------------------------
    # Histograma — punto de entrada público
    # ------------------------------------------------------------------

    def show_histogram(self):
        """Muestra el histograma según el tipo de imagen activa."""
        if not self.current_image:
            QMessageBox.warning(self.view, "Aviso", "Primero carga una imagen")
            return

        current_tab = self.view.tabs.currentIndex()

        if current_tab == 0:
            self._show_histogram_original()

        else:  # Pestaña Resultado
            if self.current_map:
                # BUG FIX: pasar image_name a get_histogram_result y get_result
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

        plt.tight_layout()
        plt.show()
        self.view.show_status("Histograma mostrado")

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
