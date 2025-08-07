import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern

import dlib

# ==== Thông số LBP ====
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

# ==== Load Dlib Predictor ====
predictor_path = r'E:\Mayhoc_ungdung\Report\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ==== Các hàm trích xuất đặc trưng ====
def extract_lbp(img):
    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return np.zeros(68 * 2)
    shape = predictor(img, rects[0])
    coords = np.array([[p.x, p.y] for p in shape.parts()]).flatten()
    return coords

def extract_hog(img):
    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False)
    return features

# ==== Hàm tiền xử lý và trích xuất đặc trưng ====
def preprocess_and_extract_features(img, method='hog+lbp+landmark', image_size=(48, 48)):
    img = cv2.resize(img, image_size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    features = []
    if 'hog' in method:
        features.append(extract_hog(img))
    if 'lbp' in method:
        features.append(extract_lbp(img))
    if 'landmark' in method:
        features.append(extract_landmarks(img))

    return np.concatenate(features)

# ==== Load dữ liệu ====
def load_data(base_path, method):
    X, y = [], []
    labels = sorted(os.listdir(base_path))
    for label in labels:
        folder_path = os.path.join(base_path, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features = preprocess_and_extract_features(img, method)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y), labels

# ==== Huấn luyện & đánh giá ====
def train_and_evaluate(train_path, test_path, method_name, model_save_path=None):
    print(f"\n🔧 Đang chạy phương pháp: {method_name}")

    print("📥 Đang tải và xử lý dữ liệu huấn luyện...")
    X_train, y_train, labels = load_data(train_path, method=method_name)

    print("📥 Đang tải và xử lý dữ liệu kiểm thử...")
    X_test, y_test, _ = load_data(test_path, method=method_name)

    print("⚖️ Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("📉 Giảm chiều bằng PCA...")
    pca = PCA(n_components=0.97)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"➡️ Số chiều còn lại: {pca.n_components_}")

    print("🧠 Huấn luyện SVM...")
    clf = SVC(kernel='rbf', C=100.0, gamma='scale', class_weight='balanced')
    clf.fit(X_train_pca, y_train)

    print("🔍 Đánh giá mô hình...")
    y_pred = clf.predict(X_test_pca)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix - {method_name.upper()}")
    plt.tight_layout()
    plt.show()

    if model_save_path:
        with open(model_save_path, 'wb') as f:
            pickle.dump({'model': clf, 'scaler': scaler, 'pca': pca, 'labels': labels}, f)
        print(f"✅ Mô hình đã lưu: {model_save_path}")

# ==== Thử nghiệm các tổ hợp đặc trưng ====
train_path = r'E:\Mayhoc_ungdung\Report\train'
test_path = r'E:\Mayhoc_ungdung\Report\test'

feature_combinations = [
    'lbp',
    'landmark',
    'lbp+landmark',
    'hog+lbp',
    'hog+landmark',
    'hog+lbp+landmark'
]

for method in feature_combinations:
    model_filename = f"svm_emotion_model_{method.replace('+', '_')}.pkl"
    train_and_evaluate(train_path, test_path, method, model_save_path=model_filename)
