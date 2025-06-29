import cv2
import dlib
import base64
import os
import logging
from math import atan2, degrees
from tkinter import filedialog
from tkinter import Tk

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_and_correct_face(image, detector, predictor):
    """
    Обнаружение лиц, вычисление угла наклона и корректировка изображения.
    """
    try:
        # Попытка обнаружить лица
        faces = detector(image, 1)
        if len(faces) == 0:
            logging.warning("Лицо не найдено на изображении.")
            return image  # Возвращаем оригинальное изображение, если лица не найдены

        # Работаем с первым найденным лицом
        face = faces[0]
        landmarks = predictor(image, face)

        # Определение координат глаз
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Вычисление угла поворота
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = atan2(delta_y, delta_x)
        angle_degrees = degrees(angle)

        logging.info(f"Обнаружено лицо. Угол наклона: {angle_degrees:.2f} градусов.")

        # Корректировка изображения
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Обрезка изображения (по лицу)
        face_roi = (face.left(), face.top(), face.width(), face.height())
        cropped_image = rotated_image[face_roi[1]:face_roi[1] + face_roi[3], face_roi[0]:face_roi[0] + face_roi[2]]

        if cropped_image.size == 0:
            logging.warning("Не удалось обрезать лицо, изображение пустое.")
            return rotated_image

        logging.info("Изображение успешно обрезано по лицу.")
        return cropped_image

    except Exception as e:
        logging.error(f"Ошибка при обработке лица: {e}")
        return image

def process_image(image_path, output_width, output_height, model_path, output_format='jpg'):
    """
    Основная функция обработки изображения:
    - Сжатие до заданного размера.
    - Обнаружение и выравнивание лица.
    - Конвертация в base64.
    """
    try:
        # Проверка модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        # Инициализация моделей
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(model_path)

        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Изображение не загружено. Проверь правильность пути.")

        logging.info(f"Оригинальный размер изображения: {image.shape}")

        # Обработка лица
        processed_image = detect_and_correct_face(image, detector, predictor)

        # Сжатие изображения
        final_image = cv2.resize(processed_image, (output_width, output_height))
        logging.info(f"Изображение сжато до: {output_width}x{output_height}")

        # Сохранение обработанного изображения
        output_file = os.path.splitext(image_path)[0] + "_result.jpg"
        cv2.imwrite(output_file, final_image)
        logging.info(f"Результат сохранен в файл: {output_file}")

        # Конвертация в base64
        _, buffer = cv2.imencode(f'.{output_format}', final_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Сохранение base64 в текстовый файл
        base64_file = os.path.splitext(image_path)[0] + "_base64.txt"
        with open(base64_file, 'w') as f:
            f.write(img_base64)
        logging.info(f"Base64 строка сохранена в файл: {base64_file}")

        logging.info("Обработка завершена успешно.")
        return img_base64
    except Exception as e:
        logging.error(f"Ошибка при обработке изображения: {e}")
        return None

def select_image():
    """
    Функция для выбора изображения через графический интерфейс
    """
    root = Tk()
    root.withdraw()  # Скрыть главное окно
    file_path = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        logging.info(f"Выбрано изображение: {file_path}")
        return file_path
    else:
        logging.error("Изображение не выбрано!")
        return None

# Пример вызова функции
if __name__ == "__main__":
    IMAGE_PATH = select_image()  # Открывает окно выбора изображения
    if IMAGE_PATH:
        MODEL_PATH = "shape_predictor_68_face_landmarks.dat"  # Путь к модели
        OUTPUT_WIDTH = 800
        OUTPUT_HEIGHT = 600

        base64_image = process_image(IMAGE_PATH, OUTPUT_WIDTH, OUTPUT_HEIGHT, MODEL_PATH, output_format='jpg')
        if base64_image:
            logging.info("Изображение успешно обработано и преобразовано в base64.")
        else:
            logging.error("Обработка изображения завершилась неудачей.")
