"""Morfología Matemática — Práctica 4.

Operaciones implementadas (binaria y en niveles de gris / latticce):

  BÁSICAS
  -------
  erosion(img, kernel, iters)      — erosión
  dilation(img, kernel, iters)     — dilatación
  opening(img, kernel)             — apertura  = erosión → dilatación
  closing(img, kernel)             — cierre    = dilatación → erosión

  DERIVADAS BINARIAS
  ------------------
  boundary(img, kernel)            — frontera  = img − erosión(img)
  hit_or_miss(img, se_fg, se_bg)   — transformada Hit-or-Miss
  thinning(img, iterations)        — adelgazamiento iterativo
  skeleton(img)                    — esqueleto morfológico

  DERIVADAS EN GRISES (Latticce)
  -------------------------------
  gradient_morph(img, kernel, mode)    — gradiente morfológico (simétrico/erosión/dilatación)
  top_hat(img, kernel)                 — Top Hat  = img − apertura(img)
  bot_hat(img, kernel)                 — Bot Hat  = cierre(img) − img

Kernels predefinidos:
  kernel_rect(size)   — rectangulo NxN
  kernel_cross(size)  — cruz 3xN
  kernel_ellipse(size)— elipse NxN

Autoras: Alejandra Michelle Mateo Garcia · Leyva Triana Isis Valeria
ESCOM · IPN — Análisis de Imágenes — Mayo 2026
"""
from __future__ import annotations
import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Kernels / Elementos Estructurantes
# ══════════════════════════════════════════════════════════════════════════════

def kernel_rect(size: int = 3) -> np.ndarray:
    """Elemento estructurante rectangular NxN."""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def kernel_cross(size: int = 3) -> np.ndarray:
    """Elemento estructurante en forma de cruz."""
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))


def kernel_ellipse(size: int = 3) -> np.ndarray:
    """Elemento estructurante elíptico."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _get_kernel(shape: str, size: int) -> np.ndarray:
    """Retorna kernel por nombre de forma."""
    return {
        "rect":    kernel_rect(size),
        "cross":   kernel_cross(size),
        "ellipse": kernel_ellipse(size),
    }.get(shape, kernel_rect(size))


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises si la imagen es RGB."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Convierte gris a RGB para visualización uniforme."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _ensure_binary(img: np.ndarray) -> np.ndarray:
    """Garantiza imagen binaria uint8 (0/255) para operaciones binarias."""
    gray = _ensure_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# OPERACIONES BÁSICAS
# ══════════════════════════════════════════════════════════════════════════════

def erosion(img: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erosión: reduce regiones blancas.

    Binaria : un píxel queda en 1 solo si todos los píxeles bajo el EE son 1.
    Grises  : toma el mínimo de intensidades bajo el EE (min-pool).
    """
    gray = _ensure_gray(img)
    result = cv2.erode(gray, kernel, iterations=iterations)
    return _to_rgb(result)


def dilation(img: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilatación: expande regiones blancas.

    Binaria : un píxel queda en 1 si al menos uno bajo el EE es 1.
    Grises  : toma el máximo de intensidades bajo el EE (max-pool).
    """
    gray = _ensure_gray(img)
    result = cv2.dilate(gray, kernel, iterations=iterations)
    return _to_rgb(result)


def opening(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apertura = Erosión → Dilatación.

    Elimina ruido pequeño (píxeles blancos aislados) y suaviza contornos.
    Modo tradicional explícito para cumplir con el requisito de la práctica.
    """
    gray = _ensure_gray(img)
    eroded  = cv2.erode(gray,   kernel, iterations=1)
    result  = cv2.dilate(eroded, kernel, iterations=1)
    return _to_rgb(result)


def closing(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Cierre = Dilatación → Erosión.

    Rellena agujeros pequeños y une regiones cercanas.
    Modo tradicional explícito para cumplir con el requisito de la práctica.
    """
    gray   = _ensure_gray(img)
    dilated = cv2.dilate(gray,    kernel, iterations=1)
    result  = cv2.erode(dilated,  kernel, iterations=1)
    return _to_rgb(result)


# ══════════════════════════════════════════════════════════════════════════════
# OPERACIONES DERIVADAS — BINARIA
# ══════════════════════════════════════════════════════════════════════════════

def boundary(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Frontera morfológica = imagen − erosión(imagen).

    Extrae el contorno interior de las regiones binarias.
    Equivalente al gradiente morfológico por erosión en binario.
    """
    binary  = _ensure_binary(img)
    eroded  = cv2.erode(binary, kernel, iterations=1)
    result  = cv2.subtract(binary, eroded)
    return _to_rgb(result)


def hit_or_miss(img: np.ndarray,
                se_fg: np.ndarray | None = None,
                se_bg: np.ndarray | None = None) -> np.ndarray:
    """Transformada Hit-or-Miss.

    Detecta patrones específicos: requiere que SE_fg coincida con objetos
    (primer plano) y SE_bg con el fondo al mismo tiempo.

    Por defecto usa un EE de esquina para detectar esquinas en objetos.
    """
    binary = _ensure_binary(img)

    if se_fg is None:
        # EE foreground: detecta extremos horizontales
        se_fg = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]], dtype=np.uint8)
    if se_bg is None:
        # EE background: complementario al foreground
        se_bg = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 0, 1]], dtype=np.uint8)

    # Hit: erosión de la imagen con se_fg
    hit  = cv2.erode(binary, se_fg, iterations=1)
    # Miss: erosión del complemento con se_bg
    miss = cv2.erode(cv2.bitwise_not(binary), se_bg, iterations=1)
    result = cv2.bitwise_and(hit, miss)
    return _to_rgb(result)


def thinning(img: np.ndarray, iterations: int = 10) -> np.ndarray:
    """Adelgazamiento morfológico iterativo (Zhang-Suen).

    Reduce los objetos binarios a estructuras de 1 píxel de grosor
    preservando su conectividad y topología.
    """
    binary = _ensure_binary(img)
    # cv2.ximgproc no siempre disponible; implementación por erosión iterativa
    # con 8 EEs de dirección (approx. Zhang-Suen via Hit-or-Miss acumulado)
    thin = (binary // 255).astype(np.uint8)

    # 8 elementos estructurantes direccionales estándar para Zhang-Suen
    se_list = [
        (np.array([[0,0,0],[0,1,0],[1,1,1]], np.uint8),
         np.array([[1,1,1],[0,0,0],[0,0,0]], np.uint8)),
        (np.array([[0,0,0],[1,1,0],[0,1,0]], np.uint8),
         np.array([[0,1,1],[0,0,1],[0,0,0]], np.uint8)),
        (np.array([[1,0,0],[1,1,0],[1,0,0]], np.uint8),
         np.array([[0,0,1],[0,0,1],[0,0,1]], np.uint8)),
        (np.array([[0,1,0],[1,1,0],[0,0,0]], np.uint8),
         np.array([[0,0,0],[0,0,1],[1,1,0]], np.uint8)),
        (np.array([[1,1,1],[0,1,0],[0,0,0]], np.uint8),
         np.array([[0,0,0],[0,0,0],[1,1,1]], np.uint8)),
        (np.array([[0,1,0],[0,1,1],[0,0,0]], np.uint8),
         np.array([[0,0,0],[1,0,0],[0,1,1]], np.uint8)),
        (np.array([[0,0,1],[0,1,1],[0,0,1]], np.uint8),
         np.array([[1,0,0],[1,0,0],[1,0,0]], np.uint8)),
        (np.array([[0,0,0],[0,1,1],[0,1,0]], np.uint8),
         np.array([[1,1,0],[1,0,0],[0,0,0]], np.uint8)),
    ]

    img_b = binary.copy()
    for _ in range(iterations):
        prev = img_b.copy()
        for se_fg, se_bg in se_list:
            hm = hit_or_miss(img_b, se_fg, se_bg)
            hm_g = _ensure_gray(hm)
            img_b = cv2.subtract(img_b, hm_g)
        if np.array_equal(prev, img_b):
            break

    return _to_rgb(img_b)


def skeleton(img: np.ndarray) -> np.ndarray:
    """Esqueleto morfológico (Medial Axis Transform).

    Calcula el esqueleto como:
      S(img) = ∪_k [ E^k(img) − open(E^k(img)) ]
    donde E^k es la k-ésima erosión iterada y open es apertura con EE 3×3.

    Reduce objetos a líneas delgadas que preservan su topología.
    """
    binary = _ensure_binary(img)
    skel   = np.zeros_like(binary)
    kernel = kernel_cross(3)
    temp   = binary.copy()

    while True:
        eroded  = cv2.erode(temp, kernel)
        opened  = cv2.dilate(eroded, kernel)
        diff    = cv2.subtract(temp, opened)
        skel    = cv2.bitwise_or(skel, diff)
        temp    = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return _to_rgb(skel)


# ══════════════════════════════════════════════════════════════════════════════
# OPERACIONES EN GRISES — LATTICCE
# ══════════════════════════════════════════════════════════════════════════════

def gradient_morph(img: np.ndarray,
                   kernel: np.ndarray,
                   mode: str = "symmetric") -> np.ndarray:
    """Gradiente morfológico en imagen de grises.

    Modos:
      "symmetric"  : dilatación − erosión   (resalta todos los bordes)
      "erosion"    : imagen − erosión        (gradiente interno)
      "dilation"   : dilatación − imagen     (gradiente externo)
    """
    gray = _ensure_gray(img)
    dil  = cv2.dilate(gray, kernel)
    ero  = cv2.erode(gray,  kernel)

    if mode == "erosion":
        result = cv2.subtract(gray, ero)
    elif mode == "dilation":
        result = cv2.subtract(dil, gray)
    else:  # symmetric
        result = cv2.subtract(dil, ero)

    return _to_rgb(result)


def top_hat(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Top Hat = imagen − apertura(imagen).

    Resalta puntos brillantes más pequeños que el EE sobre un fondo oscuro.
    Útil para detectar manchas claras en fondos no uniformes.
    """
    gray   = _ensure_gray(img)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    result = cv2.subtract(gray, opened)
    return _to_rgb(result)


def bot_hat(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Bot Hat (Black Hat) = cierre(imagen) − imagen.

    Resalta puntos oscuros más pequeños que el EE sobre un fondo claro.
    Útil para detectar manchas oscuras en fondos no uniformes.
    """
    gray   = _ensure_gray(img)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    result = cv2.subtract(closed, gray)
    return _to_rgb(result)


def morph_smooth(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Suavizado morfológico = cierre(apertura(imagen)).

    Elimina ruido oscuro y claro preservando contornos importantes.
    Equivalente a aplicar apertura seguida de cierre.
    """
    gray    = _ensure_gray(img)
    opened  = cv2.morphologyEx(gray, cv2.MORPH_OPEN,  kernel)
    result  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return _to_rgb(result)


# ══════════════════════════════════════════════════════════════════════════════
# Función de conveniencia: ejecutar todo el conjunto sobre una imagen
# ══════════════════════════════════════════════════════════════════════════════

def run_all_binary(img: np.ndarray,
                   kernel: np.ndarray) -> dict[str, np.ndarray]:
    """Aplica todas las operaciones binarias y retorna dict de resultados."""
    return {
        "Erosión":            erosion(img,   kernel),
        "Dilatación":         dilation(img,  kernel),
        "Apertura":           opening(img,   kernel),
        "Cierre":             closing(img,   kernel),
        "Frontera":           boundary(img,  kernel),
        "Hit-or-Miss":        hit_or_miss(img),
        "Adelgazamiento":     thinning(img),
        "Esqueleto":          skeleton(img),
    }


def run_all_gray(img: np.ndarray,
                 kernel: np.ndarray) -> dict[str, np.ndarray]:
    """Aplica todas las operaciones en grises y retorna dict de resultados."""
    return {
        "Erosión":                erosion(img,  kernel),
        "Dilatación":             dilation(img, kernel),
        "Apertura":               opening(img,  kernel),
        "Cierre":                 closing(img,  kernel),
        "Gradiente simétrico":    gradient_morph(img, kernel, "symmetric"),
        "Gradiente por erosión":  gradient_morph(img, kernel, "erosion"),
        "Gradiente por dilatación": gradient_morph(img, kernel, "dilation"),
        "Top Hat":                top_hat(img,  kernel),
        "Bot Hat":                bot_hat(img,  kernel),
        "Suavizado morfológico":  morph_smooth(img, kernel),
    }
