from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ImageModel:
    path: Optional[str] = None
    image_bgr: Optional[np.ndarray] = None

    def set_image(self, path: str, image_bgr: np.ndarray) -> None:
        self.path = path
        self.image_bgr = image_bgr

    def has_image(self) -> bool:
        return self.image_bgr is not None