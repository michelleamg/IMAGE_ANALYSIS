"""Modelo de procesamiento para Prácticas 3-a, 3-b y 3-c.
- Práctica 3-a : Ruido Gaussiano y Sal & Pimienta
- Práctica 3-b : Operaciones aritméticas, lógicas y relacionales
- Práctica 3-c : Etiquetado de componentes conexas (vecindad 4 y 8)

Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria
ESCOM · IPN — Análisis de Imágenes — Marzo 2026
"""
import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 3-a  RUIDO
# ══════════════════════════════════════════════════════════════════════════════

def add_salt_pepper(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """Añade ruido sal y pimienta a una imagen en grises o RGB.

    Args:
        image  : array uint8 (H, W) o (H, W, 3).
        amount : fracción de píxeles afectados (0.0–1.0).
    Returns:
        Imagen con ruido, mismo dtype y forma que la entrada.
    """
    out = image.copy()
    h, w = image.shape[:2]
    n = int(amount * h * w)

    coords_r = np.random.randint(0, h, n)
    coords_c = np.random.randint(0, w, n)
    mask_sal = np.random.rand(n) >= 0.5   # True → sal (blanco)

    if image.ndim == 2:
        out[coords_r[mask_sal],  coords_c[mask_sal]]  = 255
        out[coords_r[~mask_sal], coords_c[~mask_sal]] = 0
    else:
        out[coords_r[mask_sal],  coords_c[mask_sal]]  = [255, 255, 255]
        out[coords_r[~mask_sal], coords_c[~mask_sal]] = [0, 0, 0]
    return out


def add_gaussian_noise(image: np.ndarray,
                       mean: float = 0.0,
                       sigma: float = 20.0) -> np.ndarray:
    """Añade ruido gaussiano a una imagen en grises o RGB.

    Args:
        image : array uint8 (H, W) o (H, W, 3).
        mean  : media del ruido (normalmente 0).
        sigma : desviación estándar; valores altos = más ruido.
    Returns:
        Imagen con ruido, clampada a [0, 255] uint8.
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Filtro de mediana para eliminar ruido sal y pimienta.

    Args:
        image : array uint8.
        ksize : tamaño del kernel (debe ser impar); recomendado 3 o 5.
    """
    return cv2.medianBlur(image, ksize)


def apply_gaussian_filter(image: np.ndarray,
                          ksize: int = 5,
                          sigma: float = 1.0) -> np.ndarray:
    """Filtro gaussiano suavizador.

    Args:
        image : array uint8.
        ksize : tamaño del kernel (impar); 5 es un buen punto de partida.
        sigma : desviación estándar del kernel.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


# ══════════════════════════════════════════════════════════════════════════════
# 3-b  OPERACIONES ARITMÉTICAS, LÓGICAS Y RELACIONALES
# ══════════════════════════════════════════════════════════════════════════════

# ── Aritméticas ───────────────────────────────────────────────────────────────

def arith_add_scalar(image: np.ndarray, value: int) -> np.ndarray:
    """Suma un escalar a todos los píxeles (saturación en 255)."""
    return cv2.add(image, np.full(image.shape, value, dtype=np.uint8))


def arith_subtract_scalar(image: np.ndarray, value: int) -> np.ndarray:
    """Resta un escalar a todos los píxeles (saturación en 0)."""
    return cv2.subtract(image, np.full(image.shape, value, dtype=np.uint8))


def arith_multiply_scalar(image: np.ndarray, factor: float) -> np.ndarray:
    """Multiplica todos los píxeles por un factor flotante (saturación en 255)."""
    out = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out


def arith_add_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Suma dos imágenes con saturación."""
    i1, i2 = _same_size(img1, img2)
    return cv2.add(i1, i2)


def arith_subtract_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Resta img2 de img1 con saturación (solo píxeles más claros)."""
    i1, i2 = _same_size(img1, img2)
    return cv2.subtract(i1, i2)


def arith_multiply_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Multiplica dos imágenes píxel a píxel (normalizada a [0,255])."""
    i1, i2 = _same_size(img1, img2)
    out = (i1.astype(np.float32) * i2.astype(np.float32)) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Lógicas ───────────────────────────────────────────────────────────────────

def logic_and(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """AND bit a bit: solo mantiene píxeles comunes (activos en ambas)."""
    i1, i2 = _same_size(img1, img2)
    return cv2.bitwise_and(i1, i2)


def logic_or(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """OR bit a bit: mantiene cualquier píxel activo en alguna imagen."""
    i1, i2 = _same_size(img1, img2)
    return cv2.bitwise_or(i1, i2)


def logic_xor(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """XOR bit a bit: mantiene píxeles que difieren entre las dos imágenes."""
    i1, i2 = _same_size(img1, img2)
    return cv2.bitwise_xor(i1, i2)


def logic_not(image: np.ndarray) -> np.ndarray:
    """NOT bit a bit: invierte todos los bits de cada píxel."""
    return cv2.bitwise_not(image)


# ── Relacionales ──────────────────────────────────────────────────────────────

def relational_greater(image: np.ndarray, threshold: int) -> np.ndarray:
    """Máscara binaria: píxeles con intensidad > umbral → 255, resto → 0."""
    gray = _to_gray(image)
    mask = (gray > threshold).astype(np.uint8) * 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def relational_less(image: np.ndarray, threshold: int) -> np.ndarray:
    """Máscara binaria: píxeles con intensidad < umbral → 255, resto → 0."""
    gray = _to_gray(image)
    mask = (gray < threshold).astype(np.uint8) * 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def relational_equal(image: np.ndarray,
                     threshold: int,
                     tolerance: int = 10) -> np.ndarray:
    """Máscara binaria: píxeles con intensidad ≈ umbral (±tolerancia) → 255."""
    gray = _to_gray(image)
    mask = (np.abs(gray.astype(np.int16) - threshold) <= tolerance).astype(np.uint8) * 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# 3-c  ETIQUETADO DE COMPONENTES CONEXAS
# ══════════════════════════════════════════════════════════════════════════════

def connected_components(binary_rgb: np.ndarray,
                         connectivity: int = 8):
    """Etiquetado de componentes conexas sobre imagen binaria RGB/gris.

    Args:
        binary_rgb   : imagen RGB uint8 (se convierte internamente a binario).
        connectivity : 4 = solo ortogonal, 8 = incluye diagonales.

    Returns:
        dict con:
          "num_objects" : int — número de objetos detectados (sin contar fondo)
          "labels_img"  : np.ndarray uint8 RGB coloreado por etiqueta
          "contours_img": np.ndarray uint8 RGB con contornos verdes y números
          "stats"       : lista de dicts por objeto {"id", "area", "cx", "cy",
                                                      "x","y","w","h"}
    """
    binary_gray = _to_binary(binary_rgb)

    num_labels, labels, stats_cv, centroids = cv2.connectedComponentsWithStats(
        binary_gray, connectivity=connectivity
    )
    num_objects = num_labels - 1  # excluir fondo (etiqueta 0)

    # ── Imagen coloreada por etiqueta ──────────────────────────────────
    colored = _colorize_labels(labels, num_labels)

    # ── Imagen con contornos y numeración ─────────────────────────────
    contours_img = cv2.cvtColor(binary_gray, cv2.COLOR_GRAY2RGB).copy()
    contours_raw, _ = cv2.findContours(
        binary_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    object_stats = []
    for i, cnt in enumerate(contours_raw):
        cv2.drawContours(contours_img, [cnt], -1, (74, 222, 128), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        # Número sobre el objeto
        cv2.putText(
            contours_img, str(i + 1),
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 2,
        )
        object_stats.append({
            "id": i + 1,
            "area": int(cv2.contourArea(cnt)),
            "cx": cx, "cy": cy,
            "x": x, "y": y, "w": w, "h": h,
        })

    return {
        "num_objects": num_objects,
        "connectivity": connectivity,
        "labels_img":   colored,
        "contours_img": contours_img,
        "stats":        object_stats,
    }


def compare_connectivity(binary_rgb: np.ndarray):
    """Ejecuta etiquetado con vecindad 4 y 8 y retorna ambos resultados.

    Returns:
        Tuple (result_4, result_8) donde cada uno es el dict de
        connected_components().
    """
    return (
        connected_components(binary_rgb, connectivity=4),
        connected_components(binary_rgb, connectivity=8),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers internos
# ══════════════════════════════════════════════════════════════════════════════

def _same_size(img1: np.ndarray, img2: np.ndarray):
    """Redimensiona img2 al tamaño de img1 si difieren."""
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    # Igualar número de canales
    if img1.ndim != img2.ndim:
        if img1.ndim == 2 and img2.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        elif img1.ndim == 3 and img2.ndim == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    return img1, img2


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises si es necesario."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def _to_binary(image: np.ndarray) -> np.ndarray:
    """Convierte a binario uint8 (0/255) desde RGB o gris."""
    gray = _to_gray(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def _colorize_labels(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """Asigna un color HSV distintivo a cada etiqueta."""
    h, w = labels.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    if num_labels <= 1:
        return output
    for label in range(1, num_labels):
        hue = int((label / (num_labels - 1)) * 170)  # 0–170 en espacio HSV de OpenCV
        color_hsv = np.uint8([[[hue, 220, 210]]])
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0].tolist()
        output[labels == label] = color_rgb
    return output
