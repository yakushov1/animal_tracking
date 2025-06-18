### функция для выбора точек - центров банок. Нуждается  в перепроверке output


import os
import cv2
import csv

# Глобальные переменные
current_state = "select_left"  # Состояние выбора: "select_left", "select_right", "done"
left_point = None
right_point = None
image_clone = None
current_image = None


def select_points(event, x, y, flags, param):
    global left_point, right_point, image_clone, current_image, current_state

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_state == "select_left":
            left_point = (x, y)
            current_state = "select_right"
            update_display()
            print(
                "Now select the RIGHT point (or right-click to cancel left point selection)"
            )

        elif current_state == "select_right":
            right_point = (x, y)
            current_state = "done"
            update_display()
            print(f"Points selected for {current_image}. Press any key to continue.")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_state == "select_right" and left_point:
            # Отмена выбора левой точки
            left_point = None
            current_state = "select_left"
            update_display()
            print("Left point selection canceled. Please select left point again.")


def update_display():
    global image_clone, current_image, left_point, right_point, current_state

    display_image = image_clone.copy()

    # Отображаем инструкцию на изображении
    if current_state == "select_left":
        hint = "Select LEFT point (LMB)"
    elif current_state == "select_right":
        hint = "Select RIGHT point (LMB) or cancel left (RMB)"
    else:
        hint = "Selection complete. Press any key to continue"

    cv2.putText(
        display_image, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    # Рисуем выбранные точки
    if left_point:
        cv2.circle(display_image, left_point, 5, (0, 255, 0), -1)
        cv2.putText(
            display_image,
            f"L({left_point[0]},{left_point[1]})",
            (left_point[0] + 10, left_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    if right_point:
        cv2.circle(display_image, right_point, 5, (0, 0, 255), -1)
        cv2.putText(
            display_image,
            f"R({right_point[0]},{right_point[1]})",
            (right_point[0] + 10, right_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    cv2.imshow(current_image, display_image)


def get_image_id(filename):
    """Извлекаем ID изображения из имени файла (без расширения)"""
    return os.path.splitext(filename)[0]


def process_images(folder1, folder2, output_csv):
    global left_point, right_point, image_clone, current_image, current_state

    # Получаем список файлов из обеих папок
    images = []
    for folder in [folder1, folder2]:
        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                images.append(os.path.join(folder, file))

    # Создаем CSV файл для сохранения результатов
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Новый формат заголовков
        writer.writerow(["Image ID", "XL", "YL", "XR", "YR"])

        for image_path in images:
            # Сброс состояния для нового изображения
            left_point = None
            right_point = None
            current_state = "select_left"
            current_image = os.path.basename(image_path)
            image_id = get_image_id(current_image)

            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not open {image_path}")
                continue

            image_clone = image.copy()

            # Создаем окно и устанавливаем обработчик событий
            cv2.namedWindow(current_image)
            cv2.setMouseCallback(current_image, select_points)

            # Начальное отображение
            update_display()
            print(f"\nProcessing {current_image}")
            print("Select the LEFT point (left mouse click)")

            # Основной цикл обработки
            while True:
                key = cv2.waitKey(1) & 0xFF

                # Выход по нажатию 'q'
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    print("Program terminated early")
                    return

                # Пропуск изображения по нажатию 's'
                if key == ord("s"):
                    print(f"Image {current_image} skipped")
                    # Записываем None для пропущенных изображений
                    writer.writerow([image_id, "None", "None", "None", "None"])
                    break

                # Переход к следующему изображению после выбора обеих точек
                if current_state == "done":
                    # Сохраняем результаты в новом формате
                    writer.writerow(
                        [
                            image_id,
                            left_point[0] if left_point else "None",
                            left_point[1] if left_point else "None",
                            right_point[0] if right_point else "None",
                            right_point[1] if right_point else "None",
                        ]
                    )
                    cv2.waitKey(500)  # Небольшая пауза перед переходом
                    break

            # Закрываем окно
            cv2.destroyWindow(current_image)

    print(f"\nAll points saved to {output_csv}")


if __name__ == "__main__":
    # Укажите пути к папкам с изображениями и выходному CSV файлу
    folder1 = "images/sor_ara"
    folder2 = "images/sor_cae"
    output_csv = "output/coordinates/center_coord.csv"

    print("Instructions:")
    print("1. Left click - select point")
    print("2. Right click - cancel left point selection")
    print("3. 's' - skip current image")
    print("4. 'q' - quit program early")
    print("\nOutput format: Image ID, XL, YL, XR, YR")

    process_images(folder1, folder2, output_csv)
