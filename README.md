# Обнаружение Рыб в Видео с Использованием YOLOv5

This project demonstrates how to use the YOLOv5 model to detect fish in a video file.


Этот проект демонстрирует, как использовать модель YOLOv5 для обнаружения рыб в видеофайле.

## Требования

- Python 3.8+
- OpenCV
- PyTorch

## Установка

1. **Клонировать репозиторий**:
    ```sh
    git clone https://github.com/Nanashi1Kuro/fish_detect.git
    cd fish_detect
    ```

2. **Создать и активировать виртуальное окружение** (опционально, но рекомендуется):
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # На Windows используйте `venv\Scripts\activate`
    ```

3. **Установить необходимые зависимости**:
    ```sh
    pip install -r requirements.txt
    ```

## Использование

Для запуска скрипта обнаружения рыб используйте следующую команду:

```sh
python detecting.py --model <path_to_model_weights> [--video <path_to_video>] [--output <path_to_output_video>] [--speed <playback_speed>]
```

### Аргументы

- `--model`: (Опционально) Путь к файлу весов модели YOLOv5. По умолчанию `fish_weights.pt`.
- `--video`: (Опционально) Путь к входному видеофайлу. По умолчанию `fish.mp4`.
- `--output`: (Опционально) Путь для сохранения выходного видео. По умолчанию `output.avi`.
- `--speed`: (Опционально) Скорость воспроизведения видео в кадрах в секунду. По умолчанию `20.0`.

### Пример

```sh
python detecting.py --model fish_weights.pt --video input_fish_video.mp4 --output detected_fish_output.avi --speed 1.0
```

### По умолчанию

Для запуска с параметрами по умолчанию используйте следующую команду:

```sh
python detecting.py
```
