from pydantic import BaseModel, Field
from typing import List, Optional

class DatosPropiedadNormalizados(BaseModel):
    id_plataforma_origen: str = Field(description="URL o ID de la plataforma")
    ubicacion: str = Field(description="Dirección completa de la propiedad")
    precio_usd: float = Field(description="Precio de lista en dólares")
    ingreso_anual_estimado: float = Field(description="Renta anual proyectada, si está disponible, si no 0")
    imagenes: List[str] = Field(description="Lista de URLs de imágenes extraídas")
    condado: str = Field(description="Condado al que pertenece la propiedad")
    latitud: Optional[float] = None
    longitud: Optional[float] = None
    descripcion_cruda: str = Field(description="El texto descriptivo original del anuncio")