Запуск контейнера с поддержкой GPU (при наличии nvidia-docker):

docker run --gpus all -p 8000:8000 clip-api


Тестирование API:

Перейдите в браузере по адресу http://localhost:8000/docs для доступа к автоматически сгенерированной Swagger UI.


Или протестируйте с помощью cURL:

Добавление класса:

curl -X POST "http://localhost:8000/add_class" -H "Content-Type: application/json" -d '{"description": "описание класса"}'

Предсказание (после добавления хотя бы одного класса):


curl -X POST "http://localhost:8000/predict" -F "file=@/путь/к/изображению.jpg"
