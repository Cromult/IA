# main.py
import os
import uuid
import json
from datetime import datetime
import torch
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import base64

# ==========================================
# 0. CONFIGURACIÓN DE CARPETAS DE DEBUG (AUDITORÍA)
# ==========================================
os.makedirs("debug_logs/extract", exist_ok=True)
os.makedirs("debug_logs/verify", exist_ok=True)

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE MODELOS
# ==========================================
app = FastAPI(title="FARMACIA_CENTINELA Vision API", version="1.2")

# En VM sin GPU usará CPU automáticamente.
# También puedes forzar CPU con: FORCE_CPU=true
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"

if FORCE_CPU:
    device = "cpu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if device == "cpu":
    torch.set_num_threads(os.cpu_count() or 2)

print(f"🚀 Iniciando servidor. Dispositivo: {device.upper()}")

# RUTAS A LOS MODELOS
RUTA_YOLO = "./runs/detect/Farmacia_Centinela/modelo_deteccion/weights/best.pt"

print("⏳ Cargando modelo YOLOv8...")
best_yolo = YOLO(RUTA_YOLO)

print("⏳ Cargando DINOv2...")
DINO_MODEL_NAME = os.getenv("DINO_MODEL_NAME", "facebook/dinov2-small")

processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device)
dino_model.eval()

print(f"✅ Modelos cargados exitosamente en memoria. DINO: {DINO_MODEL_NAME}")


# ==========================================
# 2. ESQUEMAS DE DATOS (DTOs)
# ==========================================
class ExtractRequest(BaseModel):
    image_url: str

class ExtractResponse(BaseModel):
    vector: List[float]

class VerifyRequest(BaseModel):
    image_url: str
    reference_vectors: List[List[float]] 
    threshold: Optional[float] = 0.85 

class VerifyResponse(BaseModel):
    verified: bool
    best_similarity: float
    best_match_index: int
    total_detections: int # 👈 Nuevo campo para que NestJS sepa cuántos envases hubo
    best_match_crop_base64: Optional[str] = None


# ==========================================
# 3. FUNCIONES AUXILIARES (Core Logic & Debugging)
# ==========================================
def descargar_imagen(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al descargar la imagen: {str(e)}")

def detectar_y_recortar_multiples(image_pil: Image.Image) -> Tuple[List[Image.Image], Optional[Image.Image]]:
    """
    Retorna: (Lista de recortes de TODAS las medicinas, Imagen original anotada con múltiples cajas)
    Si no detecta nada, retorna ([], None).
    """
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    resultados = best_yolo.predict(img_cv, conf=0.5, imgsz=640, device=device, verbose=False)
    
    if len(resultados[0].boxes) == 0:
        return [], None
        
    recortes_pil = []
    img_anotada = img_cv.copy()
    
    # Iterar por CADA medicina detectada
    for caja in resultados[0].boxes:
        x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
        confianza = float(caja.conf[0])
        
        # Obtener el Recorte exacto de esta caja
        recorte = img_cv[y1:y2, x1:x2]
        recortes_pil.append(Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)))
        
        # Dibujar la caja en la imagen maestra de auditoría
        cv2.rectangle(img_anotada, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_anotada, f"Medicina: {confianza:.2f}", (x1, max(y1 - 10, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
    anotada_pil = Image.fromarray(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB))
    
    return recortes_pil, anotada_pil

def extraer_vector_dino(imagen_recortada_pil: Image.Image) -> np.ndarray:
    inputs = processor(images=imagen_recortada_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :] 
    return vector.cpu().numpy().flatten()

def guardar_logs_visuales(endpoint: str, original: Image.Image, anotada: Optional[Image.Image], 
                          recorte: Optional[Image.Image], metadata: dict):
    """
    Guarda las imágenes y el JSON de métricas en la carpeta correspondiente.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    req_id = str(uuid.uuid4())[:8]
    prefijo = f"debug_logs/{endpoint}/{timestamp}_{req_id}"
    
    original.save(f"{prefijo}_1_original.jpg")
    if anotada:
        anotada.save(f"{prefijo}_2_deteccion_multiple.jpg")
    if recorte:
        recorte.save(f"{prefijo}_3_mejor_recorte.jpg") # Solo guardamos el recorte que hizo Match
        
    with open(f"{prefijo}_4_metricas.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


# ==========================================
# 4. ENDPOINTS REST
# ==========================================

@app.get("/health")
def health_check():
    return {"status": "ok", "device": device, "message": "Farmacia_Centinela Vision API funcionando."}


@app.post("/api/vision/extract", response_model=ExtractResponse)
def extract_vector(payload: ExtractRequest):
    img_original = descargar_imagen(payload.image_url)
    recortes_pil, anotada_pil = detectar_y_recortar_multiples(img_original)
    
    if not recortes_pil:
        guardar_logs_visuales("extract", img_original, None, None, {"error": "No se detectó ninguna medicina"})
        raise HTTPException(status_code=400, detail="No se detectó ninguna medicina en la imagen.")
        
    # En extracción para el dashboard, asumimos que se le toma foto a 1 sola pastilla clara. 
    # Tomamos el primer recorte (la detección más segura según YOLO).
    mejor_recorte = recortes_pil[0]
    vector = extraer_vector_dino(mejor_recorte)
    
    metadata = {
        "status": "success",
        "action": "Extracción de vector base",
        "total_detections": len(recortes_pil),
        "vector_length": len(vector)
    }
    guardar_logs_visuales("extract", img_original, anotada_pil, mejor_recorte, metadata)
    
    return ExtractResponse(vector=vector.tolist())


@app.post("/api/vision/verify", response_model=VerifyResponse)
def verify_medicine(payload: VerifyRequest):
    if not payload.reference_vectors:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un vector de referencia.")

    # 1. Descargar y procesar
    img_paciente = descargar_imagen(payload.image_url)
    recortes_paciente, anotada_paciente = detectar_y_recortar_multiples(img_paciente)
    
    if not recortes_paciente:
        guardar_logs_visuales("verify", img_paciente, None, None, {"error": "No se detectó medicina"})
        raise HTTPException(status_code=400, detail="No se detectó la medicina en la foto del paciente.")
        
    # 2. Bucle Maestro
    global_mejor_similitud = 0.0
    global_mejor_indice = -1
    mejor_recorte_match = None
    
    for recorte in recortes_paciente:
        vector_paciente = extraer_vector_dino(recorte)
        vector_paciente_2d = vector_paciente.reshape(1, -1)
        
        for i, vec_ref_list in enumerate(payload.reference_vectors):
            vec_ref_np = np.array(vec_ref_list).reshape(1, -1)
            similitud = cosine_similarity(vector_paciente_2d, vec_ref_np)[0][0]
            
            if similitud > global_mejor_similitud:
                global_mejor_similitud = float(similitud)
                global_mejor_indice = i
                mejor_recorte_match = recorte 
            
    # 3. Determinar el pase
    verificado = global_mejor_similitud >= payload.threshold
    
    # 4. ⚡ Novedad: Convertir el mejor recorte a Base64 para NestJS
    crop_b64 = None
    if mejor_recorte_match:
        buffered = BytesIO()
        mejor_recorte_match.save(buffered, format="JPEG")
        # Convertimos los bytes a string base64
        crop_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 5. Guardar Auditoría
    metadata = {
        "verified": bool(verificado),
        "best_similarity": global_mejor_similitud,
        "threshold": payload.threshold,
        "best_match_index": global_mejor_indice,
        "total_items_in_photo": len(recortes_paciente),
        "total_references_compared": len(payload.reference_vectors)
    }
    
    guardar_logs_visuales("verify", img_paciente, anotada_paciente, mejor_recorte_match, metadata)
    
    return VerifyResponse(
        verified=verificado,
        best_similarity=global_mejor_similitud,
        best_match_index=global_mejor_indice,
        total_detections=len(recortes_paciente),
        best_match_crop_base64=crop_b64 # 👈 Se lo enviamos al backend
    )