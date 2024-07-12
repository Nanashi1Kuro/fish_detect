import cv2
import torch
import logging
import argparse

def setup_logging():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    # Загрузка модели YOLOv5
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, _verbose=False)
        logging.info("Модель YOLOv5 загружена.")
        return model
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        raise

def load_video(video_path, output_path):
    # Загрузка видеофайла и создание объекта для записи выходного видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видеофайл {video_path}")
        return None, None

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    logging.info(f"Видео {video_path} загружено и готово для обработки.")
    return cap, out

def detect_objects(frame, model):
    # Обнаружение объектов (рыб) на кадре с помощью модели YOLOv5s
    results = model(frame)
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if conf > 0.5 and model.names[int(cls)] == 'fish':  # Проверка класса
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            label = f"Fish {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            logging.info(f"Обнаружена рыбка с вероятностью {conf:.2f} на координатах ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}).")
    return frame

def process_video(cap, out, model, playback_speed):
    # Обработка видео: чтение кадров, обнаружение объектов и запись обработанных кадров
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame, model)
        out.write(frame)
        cv2.imshow('Frame', frame)

        # Управление скоростью воспроизведения видео
        if playback_speed > 0:
            delay = int(1000 / playback_speed)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main(video_path, model_path, output_path, playback_speed):
    # Основная функция для обработки видео
    setup_logging()

    # Загрузка модели
    try:
        model = load_model(model_path)
    except Exception as e:
        logging.error(f"Не удалось загрузить модель: {e}")
        return

    # Загрузка видео и создание объекта для записи выходного видео
    cap, out = load_video(video_path, output_path)
    if cap is None or out is None:
        return

    # Обработка видео
    process_video(cap, out, model, playback_speed)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info("Обработка видео завершена.")

if __name__ == "__main__":
    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Обнаружение рыб в видео с помощью YOLOv5')
    parser.add_argument('--video', type=str, default='fish.mp4', help='Путь к видеофайлу')
    parser.add_argument('--model', type=str, default='weights/fish_weights.pt', help='Путь к файлу весов модели YOLOv5')
    parser.add_argument('--output', type=str, default='output.avi', help='Путь для сохранения выходного видео')
    parser.add_argument('--speed', type=float, default=20.0, help='Скорость воспроизведения видео (в кадрах в секунду)')
    args = parser.parse_args()

    main(args.video, args.model, args.output, args.speed)
