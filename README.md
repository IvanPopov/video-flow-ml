# VideoFlow Optical Flow Processor

Чистая реализация VideoFlow для генерации оптического потока из видео с кодированием в gamedev формате.

## Описание

Скрипт использует [VideoFlow](https://github.com/XiaoyuShi97/VideoFlow) для генерации оптического потока и создает side-by-side визуализацию:
- **Левая сторона**: Оригинальное видео (первые 1000 кадров)
- **Правая сторона**: Оптический поток в gamedev формате

## Кодирование gamedev формата

Оптический поток кодируется в RG каналы:
- Векторы потока нормализуются относительно разрешения изображения
- Значения ограничиваются диапазоном [-20, +20]
- Кодируются как: 0 = -20, 1 = +20
- R канал: горизонтальный поток
- G канал: вертикальный поток
- B канал: не используется (0)

## Установка

### 1. Клонирование VideoFlow

```bash
git clone https://github.com/XiaoyuShi97/VideoFlow.git
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Загрузка весов модели

Скачайте предобученную модель с [VideoFlow GitHub](https://github.com/XiaoyuShi97/VideoFlow):

- Скачайте `MOF_sintel.pth` (Multi-frame Optical Flow)
- Поместите файл в директорию `VideoFlow_ckpt/`

### Структура файлов

```
video-flow-ml/
├── VideoFlow/                    # Клонированный репозиторий VideoFlow
├── VideoFlow_ckpt/               # Директория с весами модели
│   └── MOF_sintel.pth           # Веса модели VideoFlow MOF
├── flow_processor.py            # Основной скрипт
├── requirements.txt             # Зависимости
└── big_buck_bunny_720p_h264.mov # Входное видео
```

## Использование

### Базовое использование

```bash
python flow_processor.py --input big_buck_bunny_720p_h264.mov --output result.mp4
```

### Параметры

- `--input`: Путь к входному видео (по умолчанию: `big_buck_bunny_720p_h264.mov`)
- `--output`: Путь к выходному видео (по умолчанию: `videoflow_result.mp4`)
- `--device`: Устройство обработки (`auto`, `cuda`, `cpu`)

## Технические детали

### VideoFlow Multi-frame Optical Flow (MOF)

Используется модель MOF из VideoFlow, которая:
- Анализирует последовательности из 5 кадров
- Обрабатывает только первые 1000 кадров видео
- Генерирует плотный оптический поток высокого качества

### Gamedev кодирование

```python
# Нормализация относительно размера изображения
normalized_flow[:, :, 0] /= image_width   # Горизонтальная компонента
normalized_flow[:, :, 1] /= image_height  # Вертикальная компонента

# Масштабирование и ограничение до [-20, +20]
scaled_flow = normalized_flow * 200
clamped_flow = np.clip(scaled_flow, -20, 20)

# Кодирование в [0, 1]: 0 = -20, 1 = +20
encoded_flow = (clamped_flow + 20) / 40

# Сохранение в RG каналах
rgb_image[:, :, 0] = encoded_flow[:, :, 0]  # R: горизонтальный поток
rgb_image[:, :, 1] = encoded_flow[:, :, 1]  # G: вертикальный поток
rgb_image[:, :, 2] = 0.0                    # B: не используется
```

## Требования

- Python 3.7+
- PyTorch 1.6.0+
- CUDA (рекомендуется для GPU ускорения)
- VideoFlow репозиторий
- Предобученные веса MOF_sintel.pth

## Поддерживаемые форматы

- **Входные**: MP4, MOV, AVI и другие форматы видео
- **Выходные**: MP4

## Примеры результатов

Выходное видео содержит:

- **Левая панель**: Оригинальное видео (первые 1000 кадров)
- **Правая панель**: Визуализация оптического потока:
  - Черные пиксели = отсутствие движения
  - Красные оттенки = движение вправо
  - Зеленые оттенки = движение вниз
  - Желтые оттенки = диагональное движение

## Ссылки

- [VideoFlow GitHub](https://github.com/XiaoyuShi97/VideoFlow)
- [VideoFlow Paper (ICCV 2023)](https://arxiv.org/abs/2303.08340) 