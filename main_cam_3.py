#!source venv/bin/activate
import cv2
from tracker2 import * # Ваш існуючий трекер
import time
from datetime import datetime, timezone
from picamera2 import Picamera2
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import numpy as np
import math

# ============================================================================
# ### 1. ПАРАМЕТРИ ДЛЯ ФАЙНТЮНІНГУ (FINETUNING) ###
# ============================================================================

# --- Параметри детектора (з вашого коду) ---
FRAME_WIDTH = 320  # Ширина кадру "lores"
FRAME_HEIGHT = 240 # Висота кадру "lores"
ROI_RECT = {'x1': 1, 'y1': 1, 'x2': 320, 'y2': 220} # Ваша ROI [y1:y2, x1:x2]
HISTORY_FRAMES = 300
VAR_THRESHOLD = 5
CONTOUR_AREA_THRS = 200
MAX_CONTOUR_AREA = 8000 # Ваш ліміт

# --- Параметри процесора трекінгу ---
# Як довго чекати (в секундах), перш ніж вважати трек "завершеним"
MAX_TRACK_AGE_SEC = 1.5 
# Наскільки близько (в пікселях) до краю кадру має з'явитися об'єкт,
# щоб вважатися "справжнім" ("alocate" = True)
EDGE_BUFFER_PX = 15 
# Макс. відстань (в пікселях) для "склеювання" нового "false" треку 
# з існуючим "true" треком (ваша ідея "alocate")
MERGE_DISTANCE_THRESHOLD_PX = 75 

# --- Параметри для розрахунків ---
# ! КРИТИЧНО ВАЖЛИВО ! 
# Встановіть, скільки метрів в одному пікселі на вашому ROI.
PIXELS_TO_METERS = 0.05 # Припущення, потрібне калібрування!

# Класифікація на основі СЕРЕДНЬОЇ ПЛОЩІ (Area) контуру
CLASSIFICATION_THRESHOLDS = {
    'moto': 350.0,
    'car': 1500.0,
    'truck': 5000.0 
}
# Ідентифікатор камери
SENSOR_ID = "pi_cam_01"

# ============================================================================
# ### 2. ПІДКЛЮЧЕННЯ SUPABASE ###
# ============================================================================
"""load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# ! ВАЖЛИВО: Це має бути НОВА таблиця для оброблених даних
SUPABASE_TABLE = "processed_traffic_events" # Наприклад
try:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"Підключено до Supabase. Запис у таблицю: {SUPABASE_TABLE}")
except Exception as e:
    print(f"ПОМИЛКА ПІДКЛЮЧЕННЯ SUPABASE: {e}")
    client = None # Працюємо "всуху", без бази"""

# ============================================================================
# ### 3. КЛАС УПРАВЛІННЯ ТРЕКОМ (VehicleTrack) ###
# ============================================================================

class VehicleTrack:
    """
    Зберігає повну історію одного об'єкта, що відстежується, 
    та розраховує фінальну аналітику.
    """
    def __init__(self, track_id, first_detection):
        self.track_id = track_id  # Це ID з tracker2.py
        self.timestamps = [first_detection['t']]
        # Зберігаємо (x, y) ЦЕНТРУ об'єкта
        self.positions = [self.get_center(first_detection)]
        self.sizes_area = [first_detection['area']]
        self.last_seen_time = time.time()
        self.entry_point = self.get_center(first_detection)
        
        # --- Ваша логіка "alocate" ---
        self.is_true_track = self.check_if_at_edge(first_detection)

    def get_center(self, detection):
        return (detection['x'] + detection['w'] / 2, detection['y'] + detection['h'] / 2)

    def update(self, detection):
        """Додає нові дані до цього треку."""
        self.timestamps.append(detection['t'])
        self.positions.append(self.get_center(detection))
        self.sizes_area.append(detection['area'])
        self.last_seen_time = time.time()

    def is_alive(self):
        """Перевіряє, чи трек ще "живий"."""
        return (time.time() - self.last_seen_time) < MAX_TRACK_AGE_SEC

    def check_if_at_edge(self, det):
        """Реалізація вашої ідеї: True, якщо об'єкт народився біля краю."""
        x, y = self.get_center(det)
        if (x < ROI_RECT['x1'] + EDGE_BUFFER_PX or
            x > ROI_RECT['x2'] - EDGE_BUFFER_PX or
            y < ROI_RECT['y1'] + EDGE_BUFFER_PX or
            y > ROI_RECT['y2'] - EDGE_BUFFER_PX):
            return True
        return False

    def get_last_position(self):
        return self.positions[-1]

    def calculate_summary(self):
        """
        Розраховує фінальну аналітику, коли об'єкт покинув кадр.
        ЦЕ І Є ВАШІ ВИХІДНІ ДАНІ.
        """
        if len(self.timestamps) < 2:
            print(f"Трек {self.track_id} видалено (недостатньо даних).")
            return None # Недостатньо даних для аналізу

        # 1. Час
        timestamp_entry = datetime.fromtimestamp(self.timestamps[0], tz=timezone.utc).isoformat()
        timestamp_exit = datetime.fromtimestamp(self.timestamps[-1], tz=timezone.utc).isoformat()
        duration_sec = self.timestamps[-1] - self.timestamps[0]
        
        # 2. Розмір (Класифікація)
        avg_area = np.mean(self.sizes_area)
        vehicle_class = 'unknown'
        if avg_area < CLASSIFICATION_THRESHOLDS['moto']:
            vehicle_class = 'moto'
        elif avg_area < CLASSIFICATION_THRESHOLDS['car']:
            vehicle_class = 'car'
        elif avg_area < CLASSIFICATION_THRESHOLDS['truck']:
            vehicle_class = 'truck'
        else:
            vehicle_class = 'large_truck'

        # 3. Напрямок (використовує НОВУ, спрощену логіку)
        exit_point = self.positions[-1]
        direction = self.calculate_direction(self.entry_point, exit_point)
        
        # 4. Швидкість (залишається БЕЗ ЗМІН, завжди позитивна)
        avg_speed_kmh = self.calculate_speed(duration_sec)
        
        # Перевірка на адекватність (напр., швидкість < 200 км/год)
        if avg_speed_kmh > 200:
            print(f"Трек {self.track_id} видалено (аномальна швидкість: {avg_speed_kmh} km/h).")
            return None

        return {
            "tracked_id": str(self.track_id),
            "vehicle_class": vehicle_class,
            "avg_speed_kmh": round(avg_speed_kmh, 2),
            "direction": direction,
            "timestamp_entry": timestamp_entry,
            "timestamp_exit": timestamp_exit,
            "duration_sec": round(duration_sec, 2),
            "avg_area_px": round(avg_area, 2),
            "sensor_id": SENSOR_ID
        }

    # ============================================================
    # ### 💡 ОНОВЛЕНИЙ МЕТОД 💡 ###
    # ============================================================
    def calculate_direction(self, start_pos, end_pos):
        """
        Визначає напрямок руху (East/West) на основі зміни X.
        Спеціально для колій, паралельних осі X.
        """
        # start_pos = (x1, y1), end_pos = (x2, y2)
        dx = end_pos[0] - start_pos[0]
        
        # Порогове значення, щоб ігнорувати "тремтіння" або невеликі маневри
        STATIONARY_X_THRESHOLD = 15.0 # Має зміститись хоча б на 15 пікселів
        
        if dx > STATIONARY_X_THRESHOLD:
            # "Позитивна" зміна X -> Рух вправо (збільшення X)
            return "Eastbound" # Наприклад, "На схід"
        elif dx < -STATIONARY_X_THRESHOLD:
            # "Негативна" зміна X -> Рух вліво (зменшення X)
            return "Westbound" # Наприклад, "На захід"
        else:
            # Зміни по X не було (або вона була занадто малою)
            return "Stationary" # "Стоїть на місці"
    # ============================================================
    
    def calculate_speed(self, duration_sec):
        """
        Розраховує середню швидкість у км/год.
        ЦЯ ФУНКЦІЯ ЗАЛИШАЄТЬСЯ БЕЗ ЗМІН.
        Вона рахує повну відстань і завжди повертає позитивне число.
        """
        if duration_sec == 0:
            return 0
            
        # Рахуємо загальну відстань в пікселях (по всіх сегментах)
        total_distance_px = 0
        for i in range(len(self.positions) - 1):
            # math.dist рахує sqrt(dx^2 + dy^2), що завжди позитивне
            total_distance_px += math.dist(self.positions[i], self.positions[i+1])
        
        distance_meters = total_distance_px * PIXELS_TO_METERS
        speed_mps = distance_meters / duration_sec # Метри в секунду
        speed_kmh = speed_mps * 3.6
        return speed_kmh
# ============================================================================
# ### 4. ДОПОМІЖНІ ФУНКЦІЇ ПРОЦЕСОРА ###
# ============================================================================

def find_nearest_true_track(new_track_pos, active_tracks):
    """
    Знаходить найближчий "is_true_track" до нового "false" треку.
    Реалізація вашої ідеї "alocate".
    """
    min_dist = float('inf')
    best_match_id = None
    
    for track_id, track in active_tracks.items():
        if track.is_true_track:
            dist = math.dist(new_track_pos, track.get_last_position())
            if dist < min_dist and dist < MERGE_DISTANCE_THRESHOLD_PX:
                min_dist = dist
                best_match_id = track_id
                
    return best_match_id

def send_to_supabase(summary_data):
    """Безпечно відправляє 1 рядок фінальної аналітики в Supabase."""
    if summary_data is None or client is None:
        return
    try:
        print(f"✅ ВІДПРАВКА В DB: {summary_data['tracked_id']}, Клас: {summary_data['vehicle_class']}, Швидкість: {summary_data['avg_speed_kmh']} km/h")
        client.from_(SUPABASE_TABLE).insert(summary_data).execute()
    except Exception as e:
        print(f"❌ ПОМИЛКА БАЗИ ДАНИХ: {e}")

# ============================================================================
# ### 5. ГОЛОВНИЙ ЦИКЛ ПРОГРАМИ (КАМЕРА + ОБРОБКА) ###
# ============================================================================

# --- Ініціалізація (ваш код) ---
obj = cv2.createBackgroundSubtractorMOG2(history=HISTORY_FRAMES, varThreshold=VAR_THRESHOLD)
tracker = Tracker2()
frame_counter = 0
cap = Picamera2()

# --- Словник для "живих" треків ---
active_tracks = {} # {id: VehicleTrack}

try:
    lsize = (FRAME_WIDTH, FRAME_HEIGHT)
    video_config = cap.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"},
                                                  lores={"size": lsize, "format": "YUV420"})
    cap.configure(video_config)
    cap.start()
    print("Starting full_tracker.py...")
    time.sleep(1)

    while True:
        frame = cap.capture_array("lores")
        frame_counter += 1
        
        # 1. ДЕТЕКЦІЯ (ваш код)
        roi = frame[ROI_RECT['y1']:ROI_RECT['y2'], ROI_RECT['x1']:ROI_RECT['x2']]
        mask = obj.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        current_time_sec = time.time() # Отримуємо час ОДИН раз за кадр
        for c in cnt:
            area = cv2.contourArea(c)
            # Фільтруємо контури
            if area > CONTOUR_AREA_THRS and area < MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(c)
                points.append([current_time_sec, area, x, y, w, h]) # Використовуємо єдиний час

        # 2. ОТРИМАННЯ ТРЕКІВ ВІД tracker2.py
        # `detections_from_tracker` - це список [t, area, x, y, w, h, id]
        detections_from_tracker = tracker.update(points)
        
        current_frame_tracker_ids = set()
        
        # 3. ОБРОБКА ТА РОЗПОДІЛ (ALOCATE) ТРЕКІВ
        for det_list in detections_from_tracker:
            t, area, x, y, w, h, id = det_list
            current_frame_tracker_ids.add(id)
            
            # Словник з даними детекції
            det_data = {'t': t, 'area': area, 'x': x, 'y': y, 'w': w, 'h': h, 'id': id}
            
            if id in active_tracks:
                # 1. Трек вже існує -> оновлюємо його
                active_tracks[id].update(det_data)
            else:
                # 2. Новий трек -> створюємо
                new_track = VehicleTrack(id, det_data)
                
                if not new_track.is_true_track:
                    # 3. Новий трек "false" (в середені кадру)
                    # Шукаємо, до кого б його "приклеїти"
                    nearest_true_id = find_nearest_true_track(new_track.entry_point, active_tracks)
                    
                    if nearest_true_id:
                        # 4. ЗНАЙШЛИ! "Склеюємо"
                        print(f"🌀 Склеювання: Новий {id} -> Існуючий {nearest_true_id}")
                        # Оновлюємо існуючий трек даними "помилкового"
                        active_tracks[nearest_true_id].update(det_data)
                        # Важливо: ми також додаємо ID "помилкового" треку 
                        # до 'current_frame_tracker_ids', щоб старий трек не "помер"
                        current_frame_tracker_ids.add(nearest_true_id)
                    else:
                        # 5. "False" трек, але поруч нікого. 
                        # Додаємо його як новий
                        active_tracks[id] = new_track
                else:
                    # 6. Новий "true" трек -> просто додаємо
                    # print(f"Новий 'true' трек: {id}")
                    active_tracks[id] = new_track

            # Візуалізація (ваш код)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(roi, str(id), (x, y - 1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        
        # --- ВАШ КОД ВІДПРАВКИ В SUPABASE ЗВІДСИ ВИДАЛЕНО ---

        # 4. ОЧИЩЕННЯ ТА ВІДПРАВКА В DB
        finished_track_ids = []
        for track_id, track in active_tracks.items():
            if track_id not in current_frame_tracker_ids:
                # Цього ID немає в поточному кадрі
                if not track.is_alive():
                    # Трек "помер" (не бачили MAX_TRACK_AGE_SEC секунд)
                    summary = track.calculate_summary()
                    send_to_supabase(summary) # Відправляємо фінальну аналітику
                    finished_track_ids.append(track_id)
        
        # Видаляємо завершені треки з пам'яті
        for track_id in finished_track_ids:
            del active_tracks[track_id]

        # 5. ВІДОБРАЖЕННЯ (ваш код)
        cv2.putText(roi, f"Tracks: {len(active_tracks)}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("MASK", mask)
        cv2.imshow("ROI", roi)
        cv2.imshow("FRAME", frame)
        
        if cv2.waitKey(32) & 0xFF == 27: # esc
            break
            
except Exception as e:
    print(f"❌ КРИТИЧНА ПОМИЛКА: {e}")
finally:
    # Завершуємо всі треки, що залишились, при виході
    print("\nЗавершення роботи... Відправка залишків треків...")
    for track_id, track in active_tracks.items():
        summary = track.calculate_summary()
        send_to_supabase(summary)
        
    cap.stop()
    cv2.destroyAllWindows()
    print("Скрипт зупинено.")