# выделение крышек и зон вокруг банок (по контурам, которые выделил Александр)


import os
import cv2
import json
import numpy as np
from pathlib import Path


class OvalMarker:
    def __init__(self):
        self.reset_state()

        # Настройки отображения
        self.colors = {
            "inner": (0, 0, 255),  # Красный
            "outer": (0, 255, 0),  # Зеленый
            "current": (255, 0, 0),  # Синий
            "inner_fill": (0, 0, 150),  # Темно-красный для заливки inner
        }
        self.base_line_thickness = 1
        self.base_point_radius = 3
        self.zoom_scale = 1.0
        self.base_window_size = (800, 600)
        self.pan_offset = [0, 0]
        self.temp_pan_offset = [0, 0]
        self.side = None
        self.dragging = False

    def reset_state(self):
        """Сброс состояния для нового изображения"""
        self.inner_points = []
        self.outer_points = []
        self.current_points = []
        self.current_mode = "inner"
        self.image_original = None
        self.zoom_scale = 1.0
        self.pan_offset = [0, 0]
        self.temp_pan_offset = [0, 0]
        self.dragging = False

    def get_visual_properties(self):
        """Возвращает толщину линий и размер точек с учетом масштаба"""
        line_thickness = max(1, int(self.base_line_thickness / self.zoom_scale))
        point_radius = max(1, int(self.base_point_radius / self.zoom_scale))
        return line_thickness, point_radius

    def create_instruction_layer(self, width, height):
        """Создает слой с инструкциями фиксированного размера"""
        layer = np.zeros((height, width, 3), dtype=np.uint8)

        instructions = [
            "Instructions:",
            "1. Left click - add point",
            "2. 'd' - delete last point",
            "3. 'r' - reset current oval",
            "4. ESC - exit program",
            "5. ENTER - confirm oval",
            "6. Mouse wheel - zoom",
            "7. Right drag - pan",
            f"Current: {self.current_mode} oval | Side: {self.side}",
        ]

        for i, line in enumerate(instructions):
            cv2.putText(
                layer,
                line,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                layer,
                line,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return layer

    def draw_points(self, img, points, color):
        """Рисует точки и соединяет их линиями с учетом масштаба"""
        line_thickness, point_radius = self.get_visual_properties()

        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(
                    img, tuple(points[i - 1]), tuple(points[i]), color, line_thickness
                )
            cv2.line(img, tuple(points[-1]), tuple(points[0]), color, line_thickness)

        for point in points:
            cv2.circle(img, tuple(point), point_radius, color, -1)

    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши"""
        adj_x = int(
            (x - self.pan_offset[0] - self.temp_pan_offset[0]) / self.zoom_scale
        )
        adj_y = int(
            (y - self.pan_offset[1] - self.temp_pan_offset[1]) / self.zoom_scale
        )

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([adj_x, adj_y])

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.start_pan_pos = [adj_x, adj_y]

        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
            self.pan_offset[0] += self.temp_pan_offset[0]
            self.pan_offset[1] += self.temp_pan_offset[1]
            self.temp_pan_offset = [0, 0]

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.temp_pan_offset[0] = (adj_x - self.start_pan_pos[0]) * self.zoom_scale
            self.temp_pan_offset[1] = (adj_y - self.start_pan_pos[1]) * self.zoom_scale

        elif event == cv2.EVENT_MOUSEWHEEL:
            center_x = (
                x - self.pan_offset[0] - self.temp_pan_offset[0]
            ) / self.zoom_scale
            center_y = (
                y - self.pan_offset[1] - self.temp_pan_offset[1]
            ) / self.zoom_scale

            zoom_factor = 1.1
            if flags > 0:
                self.zoom_scale *= zoom_factor
            else:
                self.zoom_scale /= zoom_factor
                self.zoom_scale = max(self.zoom_scale, 0.1)

            new_center_x = (
                x - self.pan_offset[0] - self.temp_pan_offset[0]
            ) / self.zoom_scale
            new_center_y = (
                y - self.pan_offset[1] - self.temp_pan_offset[1]
            ) / self.zoom_scale

            self.pan_offset[0] += (new_center_x - center_x) * self.zoom_scale
            self.pan_offset[1] += (new_center_y - center_y) * self.zoom_scale

    def apply_zoom_pan(self, img):
        """Применяет зум и панорамирование к изображению"""
        if (
            self.zoom_scale == 1.0
            and not any(self.pan_offset)
            and not any(self.temp_pan_offset)
        ):
            return img.copy()

        M = np.float32(
            [
                [self.zoom_scale, 0, self.pan_offset[0] + self.temp_pan_offset[0]],
                [0, self.zoom_scale, self.pan_offset[1] + self.temp_pan_offset[1]],
            ]
        )

        return cv2.warpAffine(
            img,
            M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(127, 127, 127),
        )

    def save_image_with_polygons(self, img_path, output_img_dir):
        """Сохраняет изображение с выделенными полигонами"""
        if not self.inner_points or not self.outer_points:
            return

        img = self.image_original.copy()

        # Создаем маски для полигонов
        mask_outer = np.zeros(img.shape[:2], dtype=np.uint8)
        outer_array = np.array(self.outer_points, dtype=np.int32)
        cv2.fillPoly(mask_outer, [outer_array], 255)

        mask_inner = np.zeros(img.shape[:2], dtype=np.uint8)
        inner_array = np.array(self.inner_points, dtype=np.int32)
        cv2.fillPoly(mask_inner, [inner_array], 255)

        # Создаем маску для области между outer и inner
        mask_between = cv2.subtract(mask_outer, mask_inner)

        # Заливаем inner полигон (темно-красный)
        img[mask_inner > 0] = self.colors["inner_fill"]

        # Заливаем область между outer и inner (зеленый)
        img[mask_between > 0] = self.colors["outer"]

        # Рисуем контуры
        cv2.polylines(img, [outer_array], True, self.colors["outer"], 2)
        cv2.polylines(img, [inner_array], True, self.colors["inner"], 2)

        # Добавляем к имени файла сторону (L или R)
        filename = Path(img_path).stem + f"_{self.side}.png"
        output_path = os.path.join(output_img_dir, filename)

        os.makedirs(output_img_dir, exist_ok=True)
        cv2.imwrite(output_path, img)

    def process_image(self, img_path, output_file, output_img_dir):
        """Обрабатывает одно изображение"""
        self.reset_state()

        self.image_original = cv2.imread(img_path)
        if self.image_original is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            return

        window_name = f"Image: {os.path.basename(img_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name, self.base_window_size[0], self.base_window_size[1]
        )
        cv2.setMouseCallback(window_name, self.mouse_callback)

        instruction_layer = self.create_instruction_layer(
            self.image_original.shape[1], self.image_original.shape[0]
        )

        while True:
            display_img = self.image_original.copy()

            self.draw_points(display_img, self.inner_points, self.colors["inner"])
            self.draw_points(display_img, self.outer_points, self.colors["outer"])
            self.draw_points(display_img, self.current_points, self.colors["current"])

            zoomed_img = self.apply_zoom_pan(display_img)

            instruction_area = zoomed_img[
                0 : instruction_layer.shape[0], 0 : instruction_layer.shape[1]
            ]
            mask = instruction_layer > 0
            instruction_area[mask] = instruction_layer[mask]

            cv2.imshow(window_name, zoomed_img)

            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter
                if self.current_mode == "inner" and len(self.current_points) > 2:
                    self.inner_points = self.current_points.copy()
                    self.current_points = []
                    self.current_mode = "outer"
                    instruction_layer = self.create_instruction_layer(
                        self.image_original.shape[1], self.image_original.shape[0]
                    )
                elif self.current_mode == "outer" and len(self.current_points) > 2:
                    self.outer_points = self.current_points.copy()
                    self.current_points = []
                    break

            elif key == ord("d"):
                if self.current_points:
                    self.current_points.pop()

            elif key == ord("r"):
                self.current_points = []
                if self.current_mode == "inner":
                    self.inner_points = []
                else:
                    self.outer_points = []

            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                exit()

        image_id = Path(img_path).stem
        data = {
            "ID": image_id,
            "side": self.side,
            "inner": self.inner_points,
            "outer": self.outer_points,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "a") as f:
            f.write(json.dumps(data) + "\n")

        # Сохраняем изображение с полигонами
        self.save_image_with_polygons(img_path, output_img_dir)

        cv2.destroyWindow(window_name)

    def run(self):
        """Основной цикл программы"""
        while True:
            self.side = input("Enter side (L or R): ").upper()
            if self.side in ["L", "R"]:
                break
            print("Please enter only L or R")

        image_folder = "2_input/zone_by_alexandr/sor_cae"
        output_file = "3_output/coordinates/ovals/sor_cae_ovals_right_side.json"
        output_img_dir = "3_output/images_with_polygons/sor_cae"

        if os.path.exists(output_file):
            os.remove(output_file)

        for filename in sorted(os.listdir(image_folder)):
            if filename.lower().endswith(".png"):
                img_path = os.path.join(image_folder, filename)
                print(f"Processing: {filename}")
                self.process_image(img_path, output_file, output_img_dir)

        print(f"All images processed. Data saved to {output_file}")
        print(f"Images with polygons saved to {output_img_dir}")


if __name__ == "__main__":
    marker = OvalMarker()
    marker.run()
