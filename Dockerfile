# Используем официальный образ PyTorch с поддержкой CUDA 11.7 и Python 3.10
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем git
RUN apt-get update && apt-get install -y git

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

# Запускаем приложение с помощью uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]