# выделение периметра дна (с отступом 3 см вовнутрь)

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# Константы
CM_TO_PIXELS = 6  # 1 см = 6 пикселей
SHIFT_CM = 3  # Сдвиг в см
SHIFT_PX = SHIFT_CM * CM_TO_PIXELS  # Сдвиг в пикселях

# Глобальные переменные
points = []  # Список для хранения выбранных точек
img = None  # Исходное изображение
img_copy = None  # Копия для отрисовки
current_image_path = ""  # Путь к текущему изображению
all_data = []  # Список для хранения данных всех изображений
current_step = 0  # Текущий шаг выбора точек
step_names = [
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
]  # Английские подсказки
abort_flag = False  # Флаг прерывания программы


def reset_image_display():
    """Обновляет изображение с текущими точками и инструкцией"""
    global img_copy

    # Создаем свежую копию изображения
    img_copy = img.copy()

    # Рисуем все уже выбранные точки
    for i, (x, y) in enumerate(points):
        # Явное преобразование координат в целые числа
        center = int(round(x)), int(round(y))
        cv2.circle(img_copy, center, 5, (0, 255, 0), -1)
        cv2.putText(
            img_copy,
            step_names[i],
            (int(x) + 10, int(y) + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Добавляем текущую инструкцию
    if len(points) < 4:
        instruction = (
            f"Select {step_names[len(points)]} corner (Enter=Confirm, z=Undo, q=Quit)"
        )
        cv2.putText(
            img_copy,
            instruction,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def select_points(event, x, y, flags, param):
    """Функция для выбора точек мышью"""
    global points, current_step, abort_flag

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((float(x), float(y)))  # Сохраняем как float
        reset_image_display()
        cv2.imshow(f"Select points - {os.path.basename(current_image_path)}", img_copy)


def draw_shifted_rectangle(image, shifted_points, color=(0, 0, 255), thickness=2):
    """Рисует смещенный прямоугольник на изображении"""
    # Преобразуем точки в целочисленные координаты
    int_points = [(int(round(x)), int(round(y))) for (x, y) in shifted_points]
    pts = np.array(int_points, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    cv2.polylines(
        image,
        [pts],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Рисуем точки и подписи
    for i, (x, y) in enumerate(int_points):
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.putText(
            image,
            step_names[i],
            (x + 10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def process_image(image_path, output_img_folder):
    """Обрабатывает одно изображение"""
    global img, img_copy, points, current_step, current_image_path, abort_flag

    current_image_path = image_path
    points = []
    current_step = 0
    abort_flag = False

    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: failed to load image {image_path}!")
        return None

    reset_image_display()

    # Выбор точек
    window_name = f"Select points - {os.path.basename(image_path)}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_points)

    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("Select 4 points clockwise (Enter=Confirm, z=Undo, q=Quit)")

    while True:
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Полное прерывание
            abort_flag = True
            cv2.destroyAllWindows()
            return None
        elif key == ord("z") and len(points) > 0:  # Отмена последней точки
            points.pop()
            reset_image_display()
        elif key == 13 and len(points) == 4:  # ENTER - подтверждение
            break

    if abort_flag:
        return None

    if len(points) != 4:
        print(
            f"Error: need to select exactly 4 points in {os.path.basename(image_path)}!"
        )
        return None

    # Рассчитываем центр
    center_x = sum(p[0] for p in points) / 4
    center_y = sum(p[1] for p in points) / 4

    # Создаем структуру данных только для смещенных точек
    data = {
        "ID": os.path.splitext(os.path.basename(image_path))[0],
    }

    # Смещаем точки внутрь и сохраняем только их
    point_names = ["L_top", "R_top", "R_bottom", "L_bottom"]
    shifted_points = []
    for name, (x, y) in zip(point_names, points):
        new_x, new_y = shift_point(x, y, center_x, center_y, SHIFT_PX, inward=True)
        data[f"{name}_x"] = new_x
        data[f"{name}_y"] = new_y
        shifted_points.append((new_x, new_y))

    # Создаем изображение с новым прямоугольником
    result_img = img.copy()
    draw_shifted_rectangle(result_img, shifted_points)

    # Сохраняем изображение
    output_img_path = os.path.join(
        output_img_folder, f"shifted_{os.path.basename(image_path)}"
    )
    cv2.imwrite(output_img_path, result_img)

    cv2.destroyAllWindows()
    return data


def shift_point(x, y, center_x, center_y, shift_px, inward=True):
    """Сдвигает одну точку относительно центра"""
    vec_x = x - center_x
    vec_y = y - center_y

    norm = np.sqrt(vec_x**2 + vec_y**2)
    if norm == 0:
        return x, y

    vec_x /= norm
    vec_y /= norm

    if inward:
        return x - vec_x * shift_px, y - vec_y * shift_px
    else:
        return x + vec_x * shift_px, y + vec_y * shift_px


def main(folder_path, output_csv_path, output_img_folder):
    global all_data, abort_flag

    # Создаем папку для результатов если ее нет
    os.makedirs(output_img_folder, exist_ok=True)

    # Получаем все PNG-файлы в папке
    image_paths = glob(os.path.join(folder_path, "*.png"))
    if not image_paths:
        print("No PNG images found in the folder!")
        return

    # Обрабатываем каждое изображение
    for image_path in image_paths:
        if abort_flag:
            break

        data = process_image(image_path, output_img_folder)
        if data is None:
            if abort_flag:
                print("\nProcessing aborted by user")
                break
            print(f"Skipped: {os.path.basename(image_path)}")
            continue

        all_data.append(data)
        print(f"Processed: {data['ID']}")

    # Сохраняем все данные в CSV
    if all_data and not abort_flag:
        df = pd.DataFrame(all_data)

        # Упорядочиваем столбцы согласно требованиям
        columns = [
            "ID",
            "L_top_x",
            "L_top_y",
            "R_top_x",
            "R_top_y",
            "L_bottom_x",
            "L_bottom_y",
            "R_bottom_x",
            "R_bottom_y",
        ]

        df = df[columns]
        # Сохраняем с правильной кодировкой
        df.to_csv(
            output_csv_path, index=False, float_format="%.2f", encoding="utf-8-sig"
        )
        print(f"\nResults saved to {output_csv_path}")
        print(f"Images with shifted rectangles saved to {output_img_folder}")
    elif not abort_flag:
        print("No data to save")


if __name__ == "__main__":
    # Укажите пути здесь
    FOLDER_PATH = r"2_input/images/sor_cae"  # Папка с PNG
    OUTPUT_CSV_PATH = r"3_output/coordinates/square/sor_cae.csv"  # Куда сохранить CSV
    OUTPUT_IMG_FOLDER = r"3_output/images/square/sor_cae"  # Папка для изображений

    try:
        main(FOLDER_PATH, OUTPUT_CSV_PATH, OUTPUT_IMG_FOLDER)
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        cv2.destroyAllWindows()
