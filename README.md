# Многокамерная детекция людей в магазинах

Система решает задачу одновременной детекции и трекинга людей на видеопотоках нескольких камер в магазине, при этом исключая ложные срабатывания на манекенах.

---

## Структура проекта

```
├── data/  
│   ├── calibration1.py            # калибровочный файл для камеры 1  
│   ├── calibration2.py            # калибровочный файл для камеры 2  
│   ├── …  
│   └── calibrationN.py            # содержит:  
│       • imgpoints — 2D-координаты точек в изображении  
│       • objpoints — 3D-координаты тех же точек в реальном мире  
│       • mannequin_footpoints — приближённые footpoints манекенов  
│
├── experiments/                   
│   ├── single_cam_topdown.py      # детекция на одной камере  
│   ├── 2camera_topdown.py         # эксперимент на двух камерах  
│   ├── test2.ipynb                # Jupyter-ноутбук для анализа  
│   ├── multiple_camera_topdown_kalman.py  
│   └── kalman_filtering.py        # фильтр Калмана  
├── results/ # сгенерированные выходы
│   ├── output_topdown_raw.mp4
│   ├── output_topdown_smoothed.mp4
├── src/                           
│   ├── camera_calib.py            # класс Camera: калибровка и устранение дисторсии  
│   ├── multiple_camera_topdown.py # основной скрипт детекции и агрегации  
│   ├── my_botsort.yaml            # конфиг трекера BotSort  
│   ├── reid.py                    # извлечение эмбеддингов для ре-идентификации  
│   └── result_aggregator.py       # сглаживание и объединение кластеров  
│
├── requirements.txt               # зависимости с фиксированными версиями  
└── README.md                      # этот файл  
```

---

## Запуск

```bash
python3 src/multiple_camera_topdown.py
```

По умолчанию скрипт:

1. Загружает файлы калибровки из `data/calibration*.py`.  
2. Открывает видеопотоки `data/cam*.mp4`.  
3. Выполняет детекцию людей с помощью YOLO + BotSort.  
4. Отсекает манекенов   . 
5. Трансформирует координаты footpoints в координаты реального мира.  
6. Агрегирует и связывает треки между камерами.  
7. Сохраняет результаты:
   - `output_cam<i>_annotated.mp4` — видео с детекцией для каждой камеры 
   - `output_topdown_raw.mp4` — видео вида сверху без сглаживания
   - `output_topdown_smoothed.mp4` — сглаженное видео  

---
