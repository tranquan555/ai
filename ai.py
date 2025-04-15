import os
import random
import threading
import time
from collections import deque, Counter
import numpy as np
import joblib # Sử dụng joblib để lưu/tải model sklearn và scaler

# --- Cài đặt thư viện ---
# Đảm bảo các thư viện cần thiết đã được cài đặt
try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("Đang cài đặt scikit-learn...")
    os.system("pip install scikit-learn --upgrade")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    # Kiểm tra GPU (tùy chọn)
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     print(f"TensorFlow đang sử dụng GPU: {gpus}")
    # else:
    #     print("TensorFlow đang sử dụng CPU.")
except ImportError:
    print("Đang cài đặt tensorflow...")
    os.system("pip install tensorflow --upgrade")

# Import sau khi đảm bảo đã cài đặt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Thêm Gradient Boosting
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Sử dụng StandardScaler
from sklearn.exceptions import NotFittedError

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional # Thêm BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # Thêm callbacks hữu ích
from tensorflow.keras.regularizers import l1_l2 # Thêm regularizers

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.helpers import escape_markdown

print("Các thư viện đã được import thành công!")

# --- Cấu hình và Hằng số ---
# Token bot Telegram (Lấy từ biến môi trường)
TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN") # Thêm token mặc định để dễ test hơn
if TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
    print("Cảnh báo: Sử dụng token mặc định. Vui lòng đặt biến môi trường TELEGRAM_TOKEN!")

# --- Quản lý Dữ liệu và Model ---
HISTORY_MAXLEN = 1000 # Tăng giới hạn lịch sử
DICE_MAXLEN = 1000    # Lưu điểm tổng của súc sắc (hiện ít dùng cho model)
MIN_HISTORY_FOR_TRAIN = 50  # Số lượng bản ghi lịch sử tối thiểu để bắt đầu huấn luyện
MIN_HISTORY_FOR_PREDICT = 20 # Số lượng bản ghi lịch sử tối thiểu để dự đoán
LSTM_SEQUENCE_LENGTH = 25 # Độ dài chuỗi cho LSTM (quan trọng!)

MODEL_DIR = "advanced_models" # Thư mục lưu model mới
os.makedirs(MODEL_DIR, exist_ok=True) # Tạo thư mục nếu chưa có

# Đường dẫn file
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR,"history_data.npy")
DICE_FILE = os.path.join(DATA_DIR,"dice_data.npy")
SCALER_SKLEARN_FILE = os.path.join(MODEL_DIR, "sklearn_scaler.joblib")
NB_MODEL_FILE = os.path.join(MODEL_DIR, "nb_model.joblib")
LR_MODEL_FILE = os.path.join(MODEL_DIR, "lr_model.joblib")
RF_MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.joblib")
GB_MODEL_FILE = os.path.join(MODEL_DIR, "gb_model.joblib") # File cho Gradient Boosting
LSTM_MODEL_FILE = os.path.join(MODEL_DIR, "adv_lstm_best_model.keras") # Sử dụng .keras cho model TF

# --- Biến toàn cục ---
history_data = deque(maxlen=HISTORY_MAXLEN)
dice_data = deque(maxlen=DICE_MAXLEN)
sklearn_scaler = StandardScaler() # Sử dụng StandardScaler cho Sklearn features
nb_model = GaussianNB()
lr_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', max_depth=10, min_samples_leaf=5) # Tinh chỉnh RF
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42) # Thêm GB
lstm_model = None # Sẽ load hoặc build sau

# Callbacks cho việc huấn luyện LSTM
# Lưu model tốt nhất dựa trên validation loss (nếu có validation split) hoặc loss
lstm_checkpoint = ModelCheckpoint(
    LSTM_MODEL_FILE, monitor="val_loss" if MIN_HISTORY_FOR_TRAIN > 50 else "loss", save_best_only=True,
    verbose=1, mode='min', save_weights_only=False # Lưu cả kiến trúc
)
# Dừng sớm nếu validation loss không cải thiện
early_stopping = EarlyStopping(
    monitor="val_loss" if MIN_HISTORY_FOR_TRAIN > 50 else "loss", patience=15, # Tăng patience
    verbose=1, mode='min', restore_best_weights=True # Khôi phục trọng số tốt nhất
)
# Giảm learning rate nếu không cải thiện
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss" if MIN_HISTORY_FOR_TRAIN > 50 else "loss", factor=0.2, patience=7, # Tăng patience
    verbose=1, mode='min', min_lr=1e-6 # LR tối thiểu
)

# Cờ trạng thái models
sklearn_models_ready = False
lstm_model_ready = False
training_active = False
training_lock = threading.Lock()

# --- Quản lý Lưu/Tải Dữ liệu và Model ---
def save_all_data_models():
    """Lưu lịch sử, súc sắc, scaler và tất cả các mô hình."""
    print("--- Bắt đầu Lưu Dữ liệu & Models ---")
    try:
        np.save(HISTORY_FILE, np.array(history_data))
        # np.save(DICE_FILE, np.array(dice_data)) # Lưu nếu cần
        print(f"Đã lưu {len(history_data)} mục lịch sử.")

        # Lưu models sklearn và scaler
        joblib.dump(sklearn_scaler, SCALER_SKLEARN_FILE)
        print("Scaler Sklearn đã lưu.")
        if sklearn_models_ready:
            joblib.dump(nb_model, NB_MODEL_FILE)
            joblib.dump(lr_model, LR_MODEL_FILE)
            joblib.dump(rf_model, RF_MODEL_FILE)
            joblib.dump(gb_model, GB_MODEL_FILE) # Lưu GB
            print("Các mô hình Sklearn đã lưu.")
        # Lưu LSTM model (Keras tự xử lý trong ModelCheckpoint, nhưng có thể save lần cuối ở đây)
        # if lstm_model_ready and lstm_model:
        #    lstm_model.save(LSTM_MODEL_FILE) # Đảm bảo model mới nhất được lưu (có thể ghi đè checkpoint)
        #    print("LSTM model đã được lưu (lần cuối).")

    except Exception as e:
        print(f"LỖI khi lưu dữ liệu/models: {e}")
    finally:
         print("--- Kết thúc Lưu Dữ liệu & Models ---")

def load_all_data_models():
    """Tải lịch sử, súc sắc, scaler và tất cả các mô hình."""
    global history_data, dice_data, sklearn_scaler, nb_model, lr_model, rf_model, gb_model, lstm_model, sklearn_models_ready, lstm_model_ready
    print("--- Bắt đầu Tải Dữ liệu & Models ---")
    models_loaded_count = 0
    try:
        # Tải dữ liệu
        if os.path.exists(HISTORY_FILE):
            loaded_history = np.load(HISTORY_FILE).tolist()
            # Chỉ thêm vào deque nếu nó chưa tồn tại để tránh trùng lặp khi khởi động lại nhanh
            current_set = set(list(history_data)[-len(loaded_history):]) if history_data else set()
            new_items = [item for item in loaded_history if tuple(item) not in current_set] # Giả sử history_data lưu tuple hoặc str
            history_data.extend(new_items) # Dùng extend
            print(f"Đã tải {len(loaded_history)} mục lịch sử (thêm {len(new_items)} mới).")
        # if os.path.exists(DICE_FILE): # Tải nếu cần
        #     loaded_dice = np.load(DICE_FILE).tolist()
        #     dice_data.extend(loaded_dice) # Dùng extend
        #     print(f"Đã tải {len(loaded_dice)} mục súc sắc.")

        # Tải scaler và models sklearn
        if os.path.exists(SCALER_SKLEARN_FILE):
            sklearn_scaler = joblib.load(SCALER_SKLEARN_FILE)
            print("Scaler Sklearn đã tải.")
        if os.path.exists(NB_MODEL_FILE):
            nb_model = joblib.load(NB_MODEL_FILE)
            models_loaded_count+=1
        if os.path.exists(LR_MODEL_FILE):
            lr_model = joblib.load(LR_MODEL_FILE)
            models_loaded_count+=1
        if os.path.exists(RF_MODEL_FILE):
            rf_model = joblib.load(RF_MODEL_FILE)
            models_loaded_count+=1
        if os.path.exists(GB_MODEL_FILE): # Tải GB
             gb_model = joblib.load(GB_MODEL_FILE)
             models_loaded_count+=1

        if models_loaded_count == 4: # Check all 4 sklearn models
             sklearn_models_ready = True
             print(f"Đã tải {models_loaded_count}/4 mô hình Sklearn.")
        else:
             print(f"Chưa đủ mô hình Sklearn ({models_loaded_count}/4). Cần huấn luyện lại.")

        # Tải model LSTM
        if os.path.exists(LSTM_MODEL_FILE):
            try:
                # Đặt custom_objects nếu bạn có lớp tùy chỉnh (ở đây không cần)
                lstm_model = load_model(LSTM_MODEL_FILE)
                lstm_model_ready = True
                print("LSTM model đã được tải thành công.")
            except Exception as e:
                print(f"Lỗi khi tải LSTM model từ '{LSTM_MODEL_FILE}': {e}. Cần huấn luyện lại.")
                lstm_model_ready = False
        else:
            print(f"Không tìm thấy file LSTM model tại '{LSTM_MODEL_FILE}'. Cần huấn luyện.")
            lstm_model_ready = False

    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu/model cũ. Bắt đầu với dữ liệu/model mới.")
    except Exception as e:
        print(f"LỖI nghiêm trọng khi tải dữ liệu/model: {e}")
    finally:
         print(f"--- Kết thúc Tải Dữ liệu & Models --- (Sklearn Ready: {sklearn_models_ready}, LSTM Ready: {lstm_model_ready})")


# --- Feature Engineering ---
def get_basic_patterns(history_window):
    """ Trích xuất thông tin về cầu cơ bản từ cửa sổ lịch sử """
    n = len(history_window)
    if n < 3: return 0, 0, 0 # bệt, 1-1, 2-2

    # Bệt
    bet_count = 0
    last = history_window[-1]
    for i in range(n - 1, -1, -1):
        if history_window[i] == last: bet_count += 1
        else: break

    # 1-1
    one_one_count = 0
    if n >= 2 and history_window[-1] != history_window[-2]:
       one_one_count = 2
       for i in range(n - 3, -1, -2):
          if i >= 0 and history_window[i] != history_window[i+1] and history_window[i] == history_window[i+2] :
               one_one_count += 2
          else:
              break
    elif n >= 4 and all(history_window[i] != history_window[i+1] for i in range(-4, -1)): # Tăng cường check 1-1
        one_one_count = 4

    # 2-2
    two_two_count = 0
    if n >= 4:
        p1 = history_window[-4:-2]
        p2 = history_window[-2:]
        if p1[0] == p1[1] and p2[0] == p2[1] and p1[0] != p2[0]:
            two_two_count = 4
            if n >= 6:
                 p0 = history_window[-6:-4]
                 if p0[0] == p0[1] and p0[0] != p1[0]:
                     two_two_count = 6 # Chỉ tính 2-2-2... đơn giản

    return bet_count, one_one_count // 2, two_two_count // 2 # Trả về độ dài/số cặp


def create_sklearn_features(history, lookback=15):
    """
    Tạo features phức tạp hơn cho Sklearn models.
    Sử dụng một cửa sổ lịch sử (`history`) để tạo features cho điểm *sau* cửa sổ đó.
    """
    n = len(history)
    if n < lookback:
        return None # Không đủ dữ liệu trong cửa sổ

    window = history[-lookback:]
    features = []

    # 1. Lag features (3 kết quả cuối)
    for i in range(1, 4):
        features.append(1 if window[-i] == 't' else 0)

    # 2. Tỷ lệ T/X trong cửa sổ
    tai_count = window.count('t')
    xiu_count = lookback - tai_count
    features.append(tai_count / lookback)
    features.append(xiu_count / lookback)

    # 3. Các chỉ số Cầu (Bệt, 1-1, 2-2)
    bet, one_one, two_two = get_basic_patterns(window)
    features.append(bet)
    features.append(one_one)
    features.append(two_two)

    # 4. Thay đổi tỷ lệ T/X (so sánh 5 cuối vs 10 trước đó trong cửa sổ)
    if lookback >= 15:
       ratio_last_5 = window[-5:].count('t') / 5
       ratio_prev_10 = window[-15:-5].count('t') / 10 if n >= 15 else 0
       features.append(ratio_last_5 - ratio_prev_10)
    else:
        features.append(0) # Giá trị mặc định nếu lookback nhỏ

    # 5. Tần suất xuất hiện của các bộ 2 và bộ 3 gần nhất
    if n >= 2:
       last_pair = "".join(window[-2:])
       pair_counts = Counter("".join(window[i:i+2]) for i in range(lookback-1))
       features.append(pair_counts.get(last_pair, 0) / (lookback-1))
    else: features.append(0)
    if n >= 3:
        last_triplet = "".join(window[-3:])
        triplet_counts = Counter("".join(window[i:i+3]) for i in range(lookback-2))
        features.append(triplet_counts.get(last_triplet, 0) / (lookback-2))
    else: features.append(0)


    return np.array(features) # Trả về 1D array

def prepare_training_data_sklearn(full_history, lookback=15):
    """Chuẩn bị X, y cho việc huấn luyện Sklearn."""
    X, y = [], []
    labels = [1 if result == 't' else 0 for result in full_history]

    # Cần ít nhất lookback + 1 điểm để tạo feature đầu tiên và có nhãn
    if len(full_history) < lookback + 1:
        return None, None

    for i in range(lookback, len(full_history)):
        history_window = full_history[i - lookback : i]
        features = create_sklearn_features(history_window, lookback)
        if features is not None:
            X.append(features)
            y.append(labels[i]) # Nhãn là kết quả tại thời điểm i

    if not X: return None, None
    return np.array(X), np.array(y)


def prepare_lstm_data(full_history, sequence_length=LSTM_SEQUENCE_LENGTH):
    """Chuẩn bị dữ liệu (sequence_length) -> 1 cho LSTM."""
    X, y = [], []
    # Chỉ sử dụng 0 và 1 cho LSTM inputs
    labels = [1.0 if result == 't' else 0.0 for result in full_history]

    if len(labels) <= sequence_length:
        return None, None

    for i in range(len(labels) - sequence_length):
        X.append(labels[i:i + sequence_length])
        y.append(labels[i + sequence_length])

    if not X: return None, None

    X = np.array(X).astype(np.float32) # Chuyển sang float32
    y = np.array(y).astype(np.float32)
    # Reshape X cho LSTM: [samples, time_steps, features=1]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

# --- Xây dựng Mô hình LSTM Nâng cao ---
def build_advanced_lstm_model(input_shape):
    """Xây dựng kiến trúc LSTM phức tạp hơn."""
    model = Sequential(name="Advanced_LSTM")
    # Lớp 1: Bidirectional LSTM - Học chuỗi theo cả hai chiều
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization()) # Chuẩn hóa batch
    model.add(Dropout(0.4)) # Dropout mạnh hơn

    # Lớp 2: LSTM thông thường
    model.add(LSTM(96, return_sequences=True)) # Giảm số units một chút
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Lớp 3: LSTM cuối cùng
    model.add(LSTM(64, return_sequences=False)) # return_sequences=False trước Dense
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Lớp Dense trung gian
    model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))) # Thêm ReLU và Regularization

    # Lớp Output
    model.add(Dense(1, activation='sigmoid')) # Sigmoid cho dự đoán xác suất Tài/Xỉu

    # Compile với Adam optimizer và learning rate tùy chỉnh
    optimizer = Adam(learning_rate=0.001) # Bắt đầu với LR phổ biến
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print("--- Advanced LSTM Model Summary ---")
    model.summary()
    print("---------------------------------")
    return model


# --- Huấn luyện Mô hình ---
def train_all_models(force_train=False):
    """Huấn luyện tất cả các mô hình nếu đủ dữ liệu hoặc bị ép."""
    global sklearn_models_ready, lstm_model_ready, lstm_model, sklearn_scaler, training_active

    # Chỉ chạy nếu không có huấn luyện nào đang diễn ra
    if training_lock.locked() and not force_train:
        print("Huấn luyện bị bỏ qua vì đã có tiến trình khác đang chạy.")
        return
    with training_lock:
        training_active = True
        print("\n=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN ===")
        current_full_history = list(history_data)
        n_history = len(current_full_history)
        print(f"Số lượng lịch sử hiện tại: {n_history}")

        training_performed = False # Cờ để biết có lưu model không

        # --- Huấn luyện Sklearn Models ---
        sklearn_lookback = 15 # Phải khớp với `create_sklearn_features`
        min_data_sklearn = max(MIN_HISTORY_FOR_TRAIN, sklearn_lookback + 5) # Cần đủ để tạo vài sample
        if n_history >= min_data_sklearn:
            print(f"\n[SKLEARN] Đủ dữ liệu ({n_history}/{min_data_sklearn}), bắt đầu chuẩn bị...")
            X_sk, y_sk = prepare_training_data_sklearn(current_full_history, lookback=sklearn_lookback)

            if X_sk is not None and y_sk is not None and len(X_sk) >= 10: # Cần ít nhất 10 mẫu để train ổn
                print(f"[SKLEARN] Chuẩn bị {X_sk.shape[0]} mẫu (features: {X_sk.shape[1]}). Bắt đầu Scaling và Training...")
                try:
                    sklearn_scaler.fit(X_sk) # Fit scaler chỉ trên dữ liệu train
                    X_scaled = sklearn_scaler.transform(X_sk)
                    print("[SKLEARN] Scaling hoàn thành.")

                    # Train models
                    print("[SKLEARN] Training Naive Bayes...")
                    nb_model.fit(X_scaled, y_sk)
                    print("[SKLEARN] Training Logistic Regression...")
                    lr_model.fit(X_scaled, y_sk)
                    print("[SKLEARN] Training Random Forest...")
                    rf_model.fit(X_scaled, y_sk)
                    print("[SKLEARN] Training Gradient Boosting...")
                    gb_model.fit(X_scaled, y_sk) # Train GB

                    sklearn_models_ready = True
                    training_performed = True
                    print("✅ [SKLEARN] Huấn luyện thành công!")

                except Exception as e:
                    print(f"❌ [SKLEARN] LỖI trong quá trình huấn luyện: {e}")
                    sklearn_models_ready = False # Đặt lại cờ nếu lỗi
            else:
                print("[SKLEARN] Không đủ dữ liệu hợp lệ sau khi tạo features.")
        else:
            print(f"[SKLEARN] Không đủ dữ liệu ({n_history}/{min_data_sklearn}). Bỏ qua huấn luyện.")

        # --- Huấn luyện LSTM Model ---
        min_data_lstm = max(MIN_HISTORY_FOR_TRAIN, LSTM_SEQUENCE_LENGTH + 10) # Cần seq_len + đủ sample
        if n_history >= min_data_lstm:
            print(f"\n[LSTM] Đủ dữ liệu ({n_history}/{min_data_lstm}), bắt đầu chuẩn bị...")
            X_lstm, y_lstm = prepare_lstm_data(current_full_history, sequence_length=LSTM_SEQUENCE_LENGTH)

            if X_lstm is not None and y_lstm is not None and len(X_lstm) >= 10:
                print(f"[LSTM] Chuẩn bị {X_lstm.shape[0]} chuỗi (dài {X_lstm.shape[1]}).")
                try:
                    if lstm_model is None or not isinstance(lstm_model, tf.keras.Model) or force_train:
                        print("[LSTM] Xây dựng hoặc xây dựng lại mô hình LSTM...")
                        # Input shape là (sequence_length, num_features=1)
                        input_shape = (X_lstm.shape[1], X_lstm.shape[2])
                        lstm_model = build_advanced_lstm_model(input_shape=input_shape)
                    else:
                        print("[LSTM] Sử dụng lại mô hình LSTM đã có.")


                    print("[LSTM] Bắt đầu huấn luyện (fit)...")
                    # Chia train/validation nếu có đủ dữ liệu (ví dụ > 100 samples)
                    validation_split_ratio = 0.15 if len(X_lstm) > 100 else 0.0
                    history = lstm_model.fit(
                        X_lstm, y_lstm,
                        epochs=75, # Tăng epochs, EarlyStopping sẽ quản lý
                        batch_size=16, # Giảm batch size
                        validation_split=validation_split_ratio,
                        callbacks=[lstm_checkpoint, early_stopping, reduce_lr],
                        verbose=1, # Hiển thị chi tiết quá trình huấn luyện
                        shuffle=True
                    )
                    print("[LSTM] Huấn luyện (fit) hoàn thành.")
                    # Sau khi fit xong, model tốt nhất (theo val_loss/loss) đã được khôi phục bởi EarlyStopping hoặc lưu bởi Checkpoint
                    # Load lại model tốt nhất từ checkpoint để đảm bảo đang dùng bản tốt nhất
                    if os.path.exists(LSTM_MODEL_FILE):
                         print("[LSTM] Tải lại trọng số tốt nhất từ Checkpoint...")
                         lstm_model = load_model(LSTM_MODEL_FILE) # Tải lại hoàn toàn model từ file tốt nhất
                         lstm_model_ready = True
                         training_performed = True
                         print("✅ [LSTM] Huấn luyện thành công và đã tải trọng số tốt nhất!")
                    else:
                         print("[LSTM] WARN: Không tìm thấy file checkpoint sau khi huấn luyện? Model có thể chưa phải tốt nhất.")
                         lstm_model_ready = True # Vẫn coi là ready nhưng cảnh báo
                         training_performed = True


                except Exception as e:
                    print(f"❌ [LSTM] LỖI nghiêm trọng trong quá trình huấn luyện: {e}")
                    lstm_model_ready = False # Đặt lại cờ nếu lỗi
            else:
                print("[LSTM] Không đủ dữ liệu hợp lệ sau khi tạo sequences.")
        else:
            print(f"[LSTM] Không đủ dữ liệu ({n_history}/{min_data_lstm}). Bỏ qua huấn luyện.")

        # Lưu tất cả models và dữ liệu sau khi huấn luyện (nếu có thay đổi)
        if training_performed:
            print("\nHoàn thành vòng huấn luyện, lưu trạng thái...")
            save_all_data_models()
        else:
            print("\nKhông có huấn luyện nào được thực hiện trong vòng này.")

        training_active = False
        print("=== KẾT THÚC QUÁ TRÌNH HUẤN LUYỆN ===")


# --- Dự đoán Kết hợp (Ensemble) ---
def combined_prediction(current_history):
    """
    Kết hợp dự đoán từ các mô hình đã huấn luyện.
    Sử dụng weighted averaging hoặc voting dựa trên sự sẵn sàng của models.
    """
    n_hist = len(current_history)
    print(f"\n--- Bắt đầu Dự đoán Kết hợp (Lịch sử: {n_hist} mục) ---")

    if n_hist < MIN_HISTORY_FOR_PREDICT:
        print(f"WARN: Lịch sử quá ngắn ({n_hist}/{MIN_HISTORY_FOR_PREDICT}). Dự đoán ngẫu nhiên.")
        pred = random.choice(["t", "x"])
        return pred, 50.0, 50.0 # Prediction, Prob T, Prob X

    probs_t = {} # Lưu trữ xác suất Tài từ mỗi model

    # --- Dự đoán từ Sklearn Models ---
    sklearn_lookback = 15 # Phải khớp với lúc train
    if sklearn_models_ready and n_hist >= sklearn_lookback:
        print("[Predict SKLEARN] Bắt đầu...")
        features_sk = create_sklearn_features(current_history, lookback=sklearn_lookback)
        if features_sk is not None:
            try:
                # Reshape về 2D array nếu `create_sklearn_features` trả về 1D
                if features_sk.ndim == 1:
                    features_sk = features_sk.reshape(1, -1)

                scaled_features = sklearn_scaler.transform(features_sk)

                probs_t['nb'] = nb_model.predict_proba(scaled_features)[0][1]
                probs_t['lr'] = lr_model.predict_proba(scaled_features)[0][1]
                probs_t['rf'] = rf_model.predict_proba(scaled_features)[0][1]
                probs_t['gb'] = gb_model.predict_proba(scaled_features)[0][1] # Lấy prob từ GB
                print(f"[Predict SKLEARN] Probs (T): "
                      f"NB={probs_t.get('nb', -1):.3f}, LR={probs_t.get('lr', -1):.3f}, "
                      f"RF={probs_t.get('rf', -1):.3f}, GB={probs_t.get('gb',-1):.3f}")
            except NotFittedError:
                print("[Predict SKLEARN] LỖI: Mô hình Sklearn chưa được huấn luyện.")
            except ValueError as ve:
                 print(f"[Predict SKLEARN] LỖI giá trị khi dự đoán (có thể do scaler/features): {ve}")
            except Exception as e:
                print(f"[Predict SKLEARN] LỖI không xác định: {e}")
        else:
             print("[Predict SKLEARN] Không tạo được features.")
    elif not sklearn_models_ready:
         print("[Predict SKLEARN] Bỏ qua - Models chưa sẵn sàng.")
    else: # n_hist < sklearn_lookback
         print(f"[Predict SKLEARN] Bỏ qua - Lịch sử quá ngắn ({n_hist}/{sklearn_lookback}).")

    # --- Dự đoán từ LSTM Model ---
    if lstm_model_ready and lstm_model is not None and n_hist >= LSTM_SEQUENCE_LENGTH:
        print("[Predict LSTM] Bắt đầu...")
        try:
            # Chuẩn bị input sequence cuối cùng (float32, reshape)
            labels = [1.0 if result == 't' else 0.0 for result in current_history]
            last_sequence = np.array(labels[-LSTM_SEQUENCE_LENGTH:]).astype(np.float32)
            last_sequence = last_sequence.reshape((1, LSTM_SEQUENCE_LENGTH, 1))

            # Dự đoán xác suất Tài
            lstm_prob_t = lstm_model.predict(last_sequence, verbose=0)[0][0]
            probs_t['lstm'] = float(lstm_prob_t) # Chuyển về float chuẩn Python
            print(f"[Predict LSTM] Prob (T): {probs_t['lstm']:.3f}")
        except Exception as e:
            print(f"[Predict LSTM] LỖI khi dự đoán: {e}")
    elif not lstm_model_ready:
        print("[Predict LSTM] Bỏ qua - Model chưa sẵn sàng.")
    else: # n_hist < LSTM_SEQUENCE_LENGTH
        print(f"[Predict LSTM] Bỏ qua - Lịch sử quá ngắn ({n_hist}/{LSTM_SEQUENCE_LENGTH}).")

    # --- Kết hợp các dự đoán (Simple Averaging Ensemble) ---
    valid_probs = [p for p in probs_t.values() if p is not None and 0 <= p <= 1]

    if not valid_probs:
        print("Không có dự đoán hợp lệ từ bất kỳ mô hình nào. Dự đoán ngẫu nhiên.")
        final_prob_t = random.uniform(0.45, 0.55) # Hơi nhiễu quanh 0.5
    else:
        final_prob_t = sum(valid_probs) / len(valid_probs)
        print(f"Kết hợp {len(valid_probs)} dự đoán hợp lệ. Trung bình Prob (T): {final_prob_t:.4f}")

    # --- Đưa ra kết quả cuối cùng ---
    final_prediction = "t" if final_prob_t >= 0.50 else "x" # Ngưỡng 0.5
    prob_tai = final_prob_t * 100
    prob_xiu = (1 - final_prob_t) * 100

    print(f"--- Dự đoán Cuối cùng ---")
    print(f"   -> {'TÀI' if final_prediction == 't' else 'XỈU'}")
    print(f"   -> Xác suất Tài: {prob_tai:.2f}%")
    print(f"   -> Xác suất Xỉu: {prob_xiu:.2f}%")
    print(f"--------------------------")

    return final_prediction, prob_tai, prob_xiu

# --- Phát hiện cầu (giữ nguyên hoặc cải tiến thêm nếu muốn) ---
def detect_pattern_v2(history):
    """ Phát hiện cầu phức tạp hơn (có thể tham khảo các diễn đàn) """
    n = len(history)
    if n < 5: return "Không đủ dữ liệu (<5)"
    recent = history[-15:] # Phân tích 15 mục gần nhất
    m = len(recent)
    s = "".join(recent).upper() # Chuỗi để tìm kiếm

    # Ưu tiên các cầu dài
    if m>=6 and all(recent[i]==recent[-1] for i in range(-6,0)): return f"Bệt {recent[-1].upper()} (6+)"
    if m>=8 and all(recent[i]!=recent[i+1] for i in range(-8,-1)): return "1-1 (8+)"
    if m>=8 and (s.endswith("TTXXTTXX") or s.endswith("XXTTXXTT")): return "2-2 (8+)"
    if m>=8 and (s.endswith("TTTXXXTT") or s.endswith("XXXT XXX")): return "3-3 (8+)" # Ví dụ cầu 3-3
    if m>=10 and (s.endswith("TXTXTXTXTX") or s.endswith("XTXTXTXTXT")): return "1-1 (10+)"

    # Các cầu ngắn hơn
    if m>=4 and all(recent[i]==recent[-1] for i in range(-4,0)): return f"Bệt {recent[-1].upper()} (4+)"
    if m>=6 and all(recent[i]!=recent[i+1] for i in range(-6,-1)): return "1-1 (6+)"
    if m>=6 and (s.endswith("TTXXTT") or s.endswith("XXTTXX")): return "2-2 (6+)"
    if m>=5 and (s.endswith("TTTXX") or s.endswith("XXTTT")): return "Bệt 3 ngắt 2"
    if m>=5 and (s.endswith("TXT TX") or s.endswith("XTX XT")): return "1-1 ngắt 1 xen kẽ" # Cầu nhảy

    return "Không rõ hoặc cầu ngắn"

# --- Huấn luyện nền & Lên lịch ---
def background_training_job():
    global training_active
    if training_lock.locked():
        print("Huấn luyện nền đã đang chạy, bỏ qua.")
        return
    print("\n=== Trigger Huấn luyện Nền ===")
    # Có thể thêm điều kiện chỉ train nếu có dữ liệu mới đủ nhiều
    train_all_models()

def start_background_training_thread(force=False):
    # Chạy huấn luyện trong luồng riêng
    if not training_active or force:
         print("Khởi chạy luồng huấn luyện nền...")
         training_thread = threading.Thread(target=train_all_models, args=(force,), daemon=True)
         training_thread.start()
    else:
         print("Không bắt đầu luồng huấn luyện mới vì đang có tiến trình chạy.")

def schedule_background_training(interval_minutes=45):
    """Lên lịch huấn luyện định kỳ."""
    interval_seconds = interval_minutes * 60
    print(f"Bộ lên lịch huấn luyện nền sẽ chạy mỗi {interval_minutes} phút.")

    def run_scheduler():
        time.sleep(60) # Chờ 1 phút sau khi bot khởi động hẳn
        while True:
            print(f"\n[Scheduler] Chờ {interval_minutes} phút cho lần huấn luyện nền tiếp theo...")
            time.sleep(interval_seconds)
            if not training_active:
                 print("[Scheduler] Đã đến giờ, bắt đầu huấn luyện nền...")
                 start_background_training_thread()
            else:
                print("[Scheduler] Đã đến giờ, nhưng đang có huấn luyện khác chạy. Bỏ qua.")

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()


# --- Các lệnh Bot Telegram ---

async def send_long_message(update: Update, text: str):
    """Gửi tin nhắn dài bằng cách chia nhỏ."""
    max_length = constants.MessageLimit.TEXT_LENGTH
    for i in range(0, len(text), max_length):
        await update.message.reply_text(text[i:i + max_length])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /start - Khởi động, tải dữ liệu, thông báo trạng thái."""
    load_all_data_models() # Tải mọi thứ trước
    user = update.effective_user
    await update.message.reply_text(
        f"🤖 **Bot AI Tài Xỉu Nâng Cao Xin Chào {user.mention_html()}!**\n\n"
        f"Tôi sử dụng các mô hình học máy (bao gồm LSTM) để phân tích lịch sử và dự đoán kết quả tiếp theo.\n\n"
        f"🔢 Dữ liệu hiện có: {len(history_data)} records.\n"
        f"✅ Sklearn models: {'Sẵn sàng' if sklearn_models_ready else 'Cần huấn luyện'}\n"
        f"🧠 LSTM model: {'Sẵn sàng' if lstm_model_ready else 'Cần huấn luyện'}\n\n"
        "🚨 **LƯU Ý QUAN TRỌNG:**\n"
        "   - Dự đoán chỉ mang tính tham khảo, không đảm bảo chính xác 100%.\n"
        "   - Chơi có trách nhiệm, quản lý vốn cẩn thận!\n\n"
        "Sử dụng /help để xem danh sách lệnh.",
        parse_mode='HTML'
    )
    # Tự động huấn luyện lần đầu nếu chưa có model và đủ data
    if (not sklearn_models_ready or not lstm_model_ready) and len(history_data) >= MIN_HISTORY_FOR_TRAIN:
         await update.message.reply_text("Phát hiện thiếu model và đủ dữ liệu. Bắt đầu huấn luyện nền lần đầu...")
         start_background_training_thread(force=True)

    # Khởi động bộ lên lịch chạy nền (chỉ chạy 1 lần)
    if not hasattr(context.application, '_scheduler_started'):
        schedule_background_training()
        context.application._scheduler_started = True

from telegram.helpers import escape_markdown # Import the helper

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /help - Hiển thị hướng dẫn chi tiết sử dụng MarkdownV2."""

    # Chuỗi gốc chưa escape
    help_text_raw = (
        "📖 *Hướng Dẫn Sử Dụng Bot AI Tài Xỉu Nâng Cao* 📖\n\n"
        "--- *Lệnh Chính* ---\n"
        "🔹 /predict\n"
        "   Dự đoán kết quả tiếp theo dựa trên toàn bộ lịch sử bot đang ghi nhớ.\n\n"
        "🔹 /tx <lịch sử t/x>\n"
        "   Dự đoán kết quả tiếp theo dựa trên chuỗi Tài ('t') Xỉu ('x') bạn cung cấp.\n"
        "   Ví dụ: /tx t x t t x x\n"
        "   Quan trọng: Sau dự đoán, bot sẽ hỏi bạn kết quả thực tế.\n"
        "   => Việc bạn phản hồi ĐÚNG / SAI giúp bot TỰ HỌC và cải thiện!\n\n"
        "--- Quản lý Dữ liệu ---\n"
        "🔹 /add <lịch sử t/x>\n"
        "   Thêm thủ công một chuỗi kết quả vào bộ nhớ của bot.\n"
        "   Ví dụ: /add x t t x\n"
        "   (Nên dùng feedback sau /tx thay vì lệnh này để đảm bảo chất lượng dữ liệu)\n\n"
        "🔹 /history [số lượng]\n"
        "   Xem lịch sử gần đây. Mặc định là 30.\n"
        "   Ví dụ xem 50: /history 50\n\n"
        "--- Thông tin & Quản trị ---\n"
        "🔹 /status\n"
        "   Kiểm tra trạng thái hiện tại của bot (số lượng dữ liệu, models, training).\n\n"
        "🔹 /train\n"
        "   (Quản trị viên) Buộc bot huấn luyện lại tất cả mô hình ngay lập tức với dữ liệu hiện tại. (Có thể mất thời gian)\n\n"
        "--- Nguyên tắc Vàng ---\n"
        "   🧠 Bot càng có nhiều dữ liệu CHẤT LƯỢNG (được xác nhận qua feedback), dự đoán càng có cơ sở.\n"
        "   🎰 Luôn nhớ yếu tố MAY MẮN trong Tài Xỉu.\n"
        "   💰 CHƠI CÓ TRÁCH NHIỆM!"
    )

    # Escape cho MarkdownV2
    help_text_md = escape_markdown(help_text_raw, version=2)

    try:
        await update.message.reply_text(help_text_md, parse_mode='MarkdownV2')
    except Exception as e:
        print(f"!!!!!!!! LỖI TRONG HELP_COMMAND KHI GỬI MARKDOWNV2: {e}")
        await update.message.reply_text("Đã xảy ra lỗi khi hiển thị hướng dẫn Markdown. Vui lòng thử lại.")
        
async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /predict - Dự đoán dựa trên toàn bộ lịch sử hiện có."""
    current_history = list(history_data)
    if len(current_history) < MIN_HISTORY_FOR_PREDICT:
        await update.message.reply_text(f"⚠️ Cần ít nhất {MIN_HISTORY_FOR_PREDICT} mục lịch sử để dự đoán. Hiện có: {len(current_history)}.")
        return

    await update.message.reply_text("⏳ Bắt đầu phân tích và dự đoán dựa trên lịch sử hiện tại...")
    prediction, prob_tai, prob_xiu = combined_prediction(current_history)
    pattern = detect_pattern_v2(current_history) # Sử dụng detect_pattern_v2

    await update.message.reply_text(
        f"🔮 **Dự đoán Tham Khảo:**\n\n"
        f"   🧠 Kết quả: **{'TÀI' if prediction == 't' else 'XỈU'}**\n"
        f"   📊 Tỷ lệ (Tài/Xỉu): {prob_tai:.1f}% / {prob_xiu:.1f}%\n"
        f"   📈 Phân tích cầu: {pattern}\n\n"
        f"*(Lưu ý: Đây là dự đoán dựa trên toàn bộ lịch sử bot có)*"
    )

async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /tx - Dự đoán dựa trên lịch sử người dùng và yêu cầu feedback."""
    if not context.args:
        await update.message.reply_text("⚠️ Vui lòng cung cấp lịch sử! Ví dụ: `/tx t x t x x`")
        return

    user_input_str = "".join(context.args).lower()
    # Lọc chỉ giữ lại 't' và 'x'
    user_provided_history = [char for char in user_input_str if char in ['t', 'x']]

    if not user_provided_history:
        await update.message.reply_text("⚠️ Lịch sử không hợp lệ. Chỉ dùng 't' hoặc 'x'. Ví dụ: `/tx t x t x x`")
        return

    if len(user_provided_history) < 5:
        await update.message.reply_text("⚠️ Lịch sử cung cấp quá ngắn (<5). Nên cung cấp ít nhất 5-10 kết quả gần nhất.")
        # return # Cho phép chạy với ls ngắn nhưng cảnh báo

    print(f"Nhận lệnh /tx với lịch sử: {''.join(user_provided_history)}")
    await update.message.reply_text(f"⏳ Đang phân tích lịch sử bạn cung cấp: `{''.join(user_provided_history)}`...", parse_mode='Markdown')

    # Tạo lịch sử tạm thời để dự đoán = lịch sử bot + lịch sử người dùng
    # Chỉ lấy phần đủ dùng để tránh quá dài
    combined_hist_for_predict = list(history_data)[-HISTORY_MAXLEN + len(user_provided_history):] + user_provided_history
    if len(combined_hist_for_predict) > HISTORY_MAXLEN:
         combined_hist_for_predict = combined_hist_for_predict[-HISTORY_MAXLEN:]

    prediction, prob_tai, prob_xiu = combined_prediction(combined_hist_for_predict)
    pattern = detect_pattern_v2(combined_hist_for_predict)

    # --- Tạo Callback Data Quan Trọng ---
    # Lưu lại lịch sử user cung cấp và dự đoán của bot
    # Format: "txfeedback | user_history_str | prediction"
    user_history_str_for_callback = "".join(user_provided_history)
    # Giới hạn độ dài callback data (Telegram giới hạn 64 bytes)
    max_hist_len_callback = 30 # Giới hạn lịch sử trong callback
    if len(user_history_str_for_callback) > max_hist_len_callback:
        user_history_str_for_callback = user_history_str_for_callback[-max_hist_len_callback:]

    callback_data_prefix = f"txf|{user_history_str_for_callback}|{prediction}" # txf = tx feedback

    # Kiểm tra độ dài callback data
    if len(callback_data_prefix.encode('utf-8')) > 60: # Để dư vài byte cho '|correct/wrong'
         # Cần giảm độ dài user_history_str_for_callback hơn nữa
         estimated_overhead = len("txf||c") # Overhead khoảng này
         max_allowed_hist_len = 64 - estimated_overhead - len(prediction)
         user_history_str_for_callback = user_history_str_for_callback[-max_allowed_hist_len:]
         callback_data_prefix = f"txf|{user_history_str_for_callback}|{prediction}"
         print(f"WARN: Cắt ngắn callback history xuống còn {len(user_history_str_for_callback)}")

    buttons = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"✅ Thực tế là {prediction.upper()}", callback_data=f"{callback_data_prefix}|correct"),
            InlineKeyboardButton(f"❌ Không phải {prediction.upper()}", callback_data=f"{callback_data_prefix}|wrong")
        ]
        # Có thể thêm nút "Bỏ qua" nếu muốn
        # [InlineKeyboardButton("🤷 Bỏ qua", callback_data="txf_ignore")]
    ])

    await update.message.reply_text(
        f"📊 **Dự Đoán Cho Lịch Sử Bạn Cung Cấp:**\n\n"
        f"   👉 `{user_provided_history}` => Dự đoán: **{'TÀI' if prediction == 't' else 'XỈU'}**\n"
        f"   🎯 Tỷ lệ (T/X): {prob_tai:.1f}% / {prob_xiu:.1f}%\n"
        f"   📈 Cầu (phân tích tạm thời): {pattern}\n\n"
        f"❓ **Kết quả thực tế của phiên đó là gì?**\n"
        f"   *(Chọn 1 nút bên dưới để giúp bot học hỏi)*",
        reply_markup=buttons,
        parse_mode='Markdown'
    )


async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /add - Thêm dữ liệu thủ công (ít khuyến khích)."""
    if not context.args:
        await update.message.reply_text("⚠️ Vui lòng cung cấp lịch sử t/x. Ví dụ: `/add t x x t`")
        return

    input_str = "".join(context.args).lower()
    history_to_add = [char for char in input_str if char in ['t', 'x']]

    if history_to_add:
        # Thêm vào bên phải deque
        added_count = 0
        for item in history_to_add:
            if len(history_data) < HISTORY_MAXLEN:
                history_data.append(item)
                added_count += 1
            else: # Nếu deque đã đầy, bỏ cái cũ nhất và thêm cái mới
                history_data.append(item) # Deque tự xử lý việc bỏ bên trái
                added_count += 1 # Vẫn tính là đã thêm
        await update.message.reply_text(f"✅ Đã thêm {added_count} kết quả vào lịch sử. Tổng số: {len(history_data)}.")
        # Lưu lại ngay sau khi thêm thủ công
        save_all_data_models()
        # Cân nhắc huấn luyện nếu thêm nhiều? (có thể gây tốn tài nguyên)
        # if added_count > 10 and not training_active:
        #     start_background_training_thread()
    else:
        await update.message.reply_text("⚠️ Không tìm thấy dữ liệu Tài/Xỉu (t/x) hợp lệ để thêm.")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /history - Hiển thị lịch sử."""
    count = 30 # Mặc định
    if context.args:
        try:
            count = int(context.args[0])
            if not 1 <= count <= 200: # Giới hạn xem tối đa 200
                await update.message.reply_text("⚠️ Số lượng xem phải từ 1 đến 200.")
                return
        except (ValueError, IndexError):
            await update.message.reply_text("⚠️ Số lượng không hợp lệ. Ví dụ: `/history 50`")
            return

    history_list = list(history_data)
    if not history_list:
        await update.message.reply_text("⛔ Lịch sử đang trống.")
        return

    display_count = min(count, len(history_list))
    recent_history = history_list[-display_count:]
    history_str = " ".join(item.upper() for item in recent_history) # Viết hoa T X

    msg = (
        f"📜 **Lịch sử {display_count} Kết Quả Gần Nhất** (Tổng: {len(history_list)}):\n\n"
        f"`{history_str}`"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /status - Kiểm tra trạng thái chi tiết."""
    status_msg = (
        f"📊 **Trạng Thái Bot AI Tài Xỉu** 📊\n\n"
        f"**Dữ Liệu:**\n"
        f"  - Lịch sử T/X: {len(history_data)} / {HISTORY_MAXLEN}\n"
        # f"  - Dữ liệu Súc sắc: {len(dice_data)} / {DICE_MAXLEN}\n"
        f"\n**Mô Hình:**\n"
        f"  - Sklearn (NB, LR, RF, GB): {'✅ Sẵn sàng' if sklearn_models_ready else '❌ Chưa huấn luyện / Lỗi'}\n"
        f"  - Deep Learning (LSTM): {'🧠 Sẵn sàng' if lstm_model_ready else '❌ Chưa huấn luyện / Lỗi'}\n"
        f"\n**Huấn Luyện:**\n"
        f"  - Đang huấn luyện nền: {'⏳ Có' if training_active else '🚫 Không'}\n"
        f"\n*(Các mô hình được tự động huấn luyện lại định kỳ hoặc khi có đủ dữ liệu mới được xác nhận)*"
    )
    await update.message.reply_text(status_msg)

async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /train - Buộc huấn luyện lại (chỉ admin?)."""
    # Thêm kiểm tra quyền admin nếu cần
    # admin_ids = [123456789] # Thêm ID của bạn vào đây
    # if update.effective_user.id not in admin_ids:
    #    await update.message.reply_text("⛔ Bạn không có quyền thực hiện lệnh này.")
    #    return

    if training_active:
        await update.message.reply_text("⏳ Quá trình huấn luyện khác đang chạy. Vui lòng đợi...")
        return

    await update.message.reply_text("⚙️ Đã nhận lệnh! Bắt đầu **buộc** huấn luyện lại tất cả mô hình...")
    start_background_training_thread(force=True) # Sử dụng cờ force
    await update.message.reply_text("✅ Đã khởi chạy huấn luyện nền. Theo dõi log hoặc dùng /status để kiểm tra tiến trình.")


# --- Xử lý Callback (Quan trọng cho Tự học) ---
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý feedback từ nút bấm sau lệnh /tx."""
    query = update.callback_query
    await query.answer() # Thông báo đã nhận

    # Format data: "txf | user_history_str | prediction | result(correct/wrong)"
    try:
        prefix, hist_str, prediction, result = query.data.split("|")

        if prefix == "txf":
            print(f"Callback Feedback: History='{hist_str}', Prediction='{prediction}', Result='{result}'")

            if result == "correct":
                # Kết quả đúng -> Lịch sử người dùng cung cấp là chính xác VÀ kết quả SAU đó là dự đoán của bot
                validated_history = list(hist_str) # ['t', 'x', 't']
                actual_outcome = prediction         # 'x'

                # Thêm lịch sử gốc và kết quả thực tế vào data chính
                history_data.extend(validated_history)
                history_data.append(actual_outcome)
                added_count = len(validated_history) + 1

                await query.edit_message_text(
                    f"✅ Cảm ơn bạn đã xác nhận!\n"
                    f"Đã thêm {added_count} kết quả (`{hist_str} -> {actual_outcome.upper()}`) vào bộ nhớ để bot học hỏi.",
                    parse_mode='Markdown'
                )
                save_all_data_models() # Lưu ngay lập tức
                # Có thể trigger huấn luyện nếu có đủ dữ liệu mới
                # if not training_active and len(history_data) % 20 == 0: # Ví dụ train lại mỗi 20 records mới
                #     start_background_training_thread()

            elif result == "wrong":
                # Kết quả sai -> Lịch sử người dùng cung cấp là chính xác, NHƯNG kết quả SAU đó KHÁC với dự đoán
                validated_history = list(hist_str)
                actual_outcome = 'x' if prediction == 't' else 't' # Kết quả ngược lại

                # Thêm lịch sử gốc và kết quả thực tế (đã sửa) vào data
                history_data.extend(validated_history)
                history_data.append(actual_outcome)
                added_count = len(validated_history) + 1

                await query.edit_message_text(
                     f"✅ Cảm ơn bạn đã phản hồi!\n"
                     f"Đã ghi nhận kết quả thực tế là `{actual_outcome.upper()}` (khác dự đoán) và thêm {added_count} records (`{hist_str} -> {actual_outcome.upper()}`) vào bộ nhớ.",
                     parse_mode='Markdown'
                 )
                save_all_data_models()
                # if not training_active and len(history_data) % 20 == 0:
                #     start_background_training_thread()

        # elif prefix == "txf_ignore":
        #     await query.edit_message_text("ℹ️ Đã bỏ qua phản hồi cho dự đoán này.")

        else:
            await query.edit_message_text("Lỗi: Hành động callback không hợp lệ.")
            print(f"Lỗi Callback: Prefix không đúng - {query.data}")

    except ValueError:
         print(f"Lỗi Callback: Không thể split data - {query.data}")
         await query.edit_message_text("Lỗi xử lý phản hồi (data format sai).")
    except Exception as e:
        print(f"Lỗi Callback Nghiêm trọng: {e} \nData: {query.data}")
        try:
            await query.edit_message_text("Đã xảy ra lỗi không mong muốn khi xử lý lựa chọn của bạn.")
        except Exception: pass # Bỏ qua nếu không gửi được tin nhắn lỗi

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log các lỗi và gửi tin nhắn báo lỗi nếu có thể."""
    print(f"Exception while handling an update: {context.error}")
    # traceback.print_exception(type(context.error), context.error, context.error.__traceback__) # Log chi tiết hơn nếu cần

    # Thử gửi tin nhắn lỗi cho người dùng (có thể thất bại nếu lỗi mạng)
    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            await update.effective_message.reply_text("❗️ Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.")
        except Exception as e:
            print(f"Không thể gửi tin nhắn lỗi cho người dùng: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Khởi tạo Bot AI Tài Xỉu Nâng Cao ---")
    application = ApplicationBuilder().token(TOKEN).build()

    # Thêm các handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("tx", tx))
    application.add_handler(CommandHandler("add", add))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("train", train_command))
    application.add_handler(CallbackQueryHandler(handle_callback)) # Xử lý nút bấm quan trọng
    application.add_error_handler(error_handler) # Xử lý lỗi chung

    print("\n--- Bot đã sẵn sàng lắng nghe ---")
    # Chạy bot non-blocking để cho phép các tiến trình nền (scheduler) chạy
    application.run_polling(allowed_updates=Update.ALL_TYPES)

    # Phần này sẽ không bao giờ đạt được nếu run_polling chạy mãi mãi
    print("--- Bot đang dừng ---")