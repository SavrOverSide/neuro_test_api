from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import clip
import torch
from PIL import Image
import faiss
import numpy as np
import torch.nn.functional as F

app = FastAPI()

# Определяем устройство: GPU или CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель CLIP (используется версия ViT-B/32)
model, preprocess = clip.load("ViT-B/32", device=device)

# Размерность векторов (для ViT-B/32 — 512)
dim = 512

# Создаем FAISS-индекс для поиска по inner product (при нормализации – косинусное сходство)
res = faiss.StandardGpuResources()
index_cpu = faiss.IndexFlatIP(dim)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Глобальный список для хранения описаний классов.
# Рекомендуется использовать описательные фразы на английском (например, "a photo of a cat")
class_names = []

# Модель данных для запроса /add_class
class AddClassRequest(BaseModel):
    description: str

def encode_text(text: str):
    # Если вы вводите текст на русском, возможно, стоит его перевести на английский,
    # так как CLIP лучше понимает английские описания.
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def encode_image(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

@app.post("/add_class")
async def add_class(request: AddClassRequest):
    vector = encode_text(request.description)
    # Добавляем вектор в FAISS-индекс
    index.add(np.array(vector, dtype=np.float32))
    # Сохраняем описание класса для последующего отображения
    class_names.append(request.description)
    return {"message": "Class added successfully", "class_index": len(class_names) - 1}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    image_vector = encode_image(image)
    
    # Проверка: в хранилище должны быть добавлены классы
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No classes in the vector store")
    
    # Определяем количество возвращаемых ближайших соседей
    k = min(5, index.ntotal)
    D, I = index.search(np.array(image_vector, dtype=np.float32), k)
    similarities = torch.tensor(D[0])
    
    # Применяем temperature scaling с помощью CLIP logit_scale, если он доступен.
    # В оригинальном CLIP logit_scale является обучаемым параметром (в логарифмической форме).
    # Его экспонента умножается на сходства, чтобы усилить различия.
    logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 1.0
    scaled_similarities = similarities * logit_scale

    # Вычисляем softmax для получения относительных вероятностей
    probs = F.softmax(scaled_similarities, dim=0)
    best_idx = I[0][0]
    best_prob = probs[0].item()
    
    # Можно установить пороговую величину для принятия решения.
    # Порог также можно масштабировать logit_scale-ом, если требуется.
    similarity_threshold = 0.2 * logit_scale  # значение подбирается экспериментально
    if scaled_similarities[0] < similarity_threshold:
        return {
            "predicted_class": None, 
            "probability": 0.0, 
            "message": "Image does not match any known class."
        }
    
    best_class = class_names[best_idx]
    return {"predicted_class": best_class, "probability": best_prob}


@app.post("/reset_classes")
async def reset_classes():
    global index, class_names, res, index_cpu
    # Переинициализируем FAISS-индекс
    index_cpu = faiss.IndexFlatIP(dim)
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu
    # Очищаем список классов
    class_names = []
    return {"message": "Classes reset successfully"}

@app.get("/classes")
async def get_classes():
    return {"classes": class_names}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
