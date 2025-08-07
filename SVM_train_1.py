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

# ==== Thiết lập thông số ====
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

predictor_path = r'E:\Mayhoc_ungdung\Report\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ==== Hàm trích xuất landmark ====
def extract_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return np.zeros(68*2)
    shape = predictor(img, rects[0])
    coords = np.array([[p.x, p.y] for p in shape.parts()]).flatten()
    return coords

# ==== Hàm trích xuất LBP ====
def extract_lbp(img):
    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# ==== Tiền xử lý + Trích xuất HOG + LBP + Landmark ====
def preprocess_and_extract_features(img, image_size=(48, 48)):
    img = cv2.resize(img, image_size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Trích xuất HOG
    hog_features = hog(img,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       visualize=False)

    # Trích xuất LBP
    lbp_features = extract_lbp(img)

    # Trích xuất Landmark
    landmarks = extract_landmarks(img)

    return np.concatenate((hog_features, lbp_features, landmarks))

# ==== Nạp dữ liệu từ thư mục ====
def load_data_from_folder(base_path):
    X = []
    y = []
    labels = sorted(os.listdir(base_path))
    for label in labels:
        folder_path = os.path.join(base_path, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features = preprocess_and_extract_features(img)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y), labels

# ==== Đường dẫn dữ liệu ====
train_path = r'E:\Mayhoc_ungdung\Report\train'
test_path = r'E:\Mayhoc_ungdung\Report\test'

# ==== Xử lý dữ liệu ====
print("🔄 Đang xử lý dữ liệu huấn luyện...")
X_train, y_train, label_names = load_data_from_folder(train_path)

print("🔄 Đang xử lý dữ liệu kiểm thử...")
X_test, y_test, _ = load_data_from_folder(test_path)

# ==== Chuẩn hóa dữ liệu ====
print("⚖️ Đang chuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==== PCA để giảm chiều ====
print("📉 Đang giảm chiều bằng PCA...")
pca = PCA(n_components=0.97)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✅ Số chiều sau PCA: {pca.n_components_}")

# ==== Huấn luyện mô hình SVM ====
print("🧠 Đang huấn luyện mô hình SVM...")
clf = SVC(kernel='rbf', C=100.0, gamma='scale', class_weight='balanced')
clf.fit(X_train_pca, y_train)

# ==== Dự đoán và đánh giá ====
print("🔍 Đang dự đoán và đánh giá mô hình...")
y_pred = clf.predict(X_test_pca)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ==== Confusion matrix ====
cm = confusion_matrix(y_test, y_pred, labels=label_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ==== Lưu mô hình ====
with open('svm_emotion_model_6.pkl', 'wb') as f:
    pickle.dump({
        'model': clf,
        'pca': pca,
        'scaler': scaler,
        'labels': label_names
    }, f)

print("✅ Mô hình, PCA và scaler đã được lưu vào 'svm_emotion_model_6.pkl'")
