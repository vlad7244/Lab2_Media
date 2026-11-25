import cv2
import numpy as np
import os
import datetime

# Настройка рабочей директории для IDLE (папка, где лежит main.py)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
except NameError:
    pass # Если запущено не из файла

# УМНОЕ ОПРЕДЕЛЕНИЕ ПУТЕЙ
# Если мы уже находимся внутри папки Lab2_Media, то базовый путь - это текущая папка.
# Если мы снаружи, то базовый путь - это папка Lab2_Media.
if os.path.basename(os.getcwd()) == "Lab2_Media":
    BASE_DIR = ""  # Мы уже внутри Lab2_Media
else:
    BASE_DIR = "Lab2_Media" # Мы снаружи, нужно зайти в нее

def write_log(filename, status):
    """Функция для записи лога в history.txt"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{now}] Файл: {filename} | Фильтр: Unsharp Masking + Цветокоррекция | Статус: {status}\n"

    # Собираем путь к логу
    log_path = os.path.join(BASE_DIR, "logs", "history.txt")
    
    # Страховка: если папки logs нет (например, забыли создать), программа не упадет
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "a", encoding="UTF-8") as log_file:
        log_file.write(log_line)

def is_readonly(filepath):
    return not os.access(filepath, os.W_OK)

def apply_unsharp_masking(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 3.0, gaussian, -2.0, 0)
    return sharpened

def apply_color_correction(image):
    """Правильная цветокоррекция через HSV"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Увеличиваем насыщенность на 20% (коэффициент 1.2 безопаснее)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] - 40, 0, 255)
    hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
    # Возвращаем тип uint8, иначе OpenCV выдаст ошибку или артефакты
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

def main():
    # Формируем пути с учетом BASE_DIR
    input_dir = os.path.join(BASE_DIR, "data", "input")
    output_dir = os.path.join(BASE_DIR, "data", "output")

    while True:
        print("\n" + "="*50)
        filename = input("Введите имя файла (или 'exit' для выхода): ").strip()
        
        if filename.lower() == 'exit':
            print("Завершение работы программы.")
            break

        filepath = os.path.join(input_dir, filename)

        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError

            if is_readonly(filepath):
                raise PermissionError

            image = cv2.imread(filepath)
            if image is None:
                raise IOError("Не удалось прочитать изображение")

            print("Файл успешно открыт. Выполняется обработка...")

            img_sharp = apply_unsharp_masking(image)
            img_final = apply_color_correction(img_sharp)

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "result.jpg")
            success = cv2.imwrite(output_path, img_final)

            if success:
                print(f"Готово! Результат сохранен в: {output_path}")
                write_log(filename, "Успешно")
                break 
            else:
                print("Ошибка при сохранении файла.")
                write_log(filename, "Ошибка сохранения")

        except FileNotFoundError:
            print("❌ Ошибка: Файл не найден. Попробуйте снова.")
            write_log(filename, "Ошибка: Файл не найден")

        except PermissionError:
            print(f"❌ Ошибка: Нет прав на обработку файла {filename}. Обратитесь к администратору.")
            write_log(filename, "Ошибка доступа")

        except Exception as e:
            print(f"❌ Произошла непредвиденная ошибка: {e}")
            write_log(filename, f"Неизвестная ошибка: {e}")

if __name__ == "__main__":
    main()
