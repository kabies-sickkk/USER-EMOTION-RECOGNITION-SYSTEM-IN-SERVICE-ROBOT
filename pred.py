import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
from skimage.feature import hog, local_binary_pattern
import dlib
import time

# ==== Thông số LBP ====
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

# ==== Dlib landmarks ====
predictor_path = r'E:\Mayhoc_ungdung\Report\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ==== Hàm trích xuất landmarks ====
def extract_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return np.zeros(68 * 2)
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

# ==== Hàm tiền xử lý và trích xuất đặc trưng ====
def preprocess_and_extract_features(img, image_size=(48, 48)):
    img = cv2.resize(img, image_size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    lbp_features = extract_lbp(img)
    landmarks = extract_landmarks(img)
    return np.concatenate((hog_features, lbp_features, landmarks))

# ==== Tải mô hình đã huấn luyện ====
with open(r'E:\Mayhoc_ungdung\svm_emotion_model_5.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    pca = data['pca']
    scaler = data['scaler']
    labels = data['labels']

# ==== GUI ====
class EmotionRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận Diện Cảm Xúc")
        self.root.geometry("650x600")

        self.cap = None
        self.running_camera = False

        self.mode = tk.StringVar(value="image")
        self.mode.trace_add("write", self.on_mode_change)  # Gọi khi chế độ thay đổi

        # ==== Giao diện chọn chế độ ====
        mode_frame = tk.Frame(root)
        tk.Label(mode_frame, text="Chọn chế độ nhận diện:", font=("Arial", 12)).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Ảnh từ máy", variable=self.mode, value="image", font=("Arial", 11)).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Camera", variable=self.mode, value="camera", font=("Arial", 11)).pack(side=tk.LEFT)
        mode_frame.pack(pady=10)

        # ==== Nút bắt đầu ====
        self.action_btn = tk.Button(root, text="Bắt đầu nhận diện", command=self.run_recognition, font=("Arial", 13))
        self.action_btn.pack(pady=10)

        # ==== Hiển thị ảnh và kết quả ====
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16), fg="blue")
        self.result_label.pack(pady=10)

    def on_mode_change(self, *args):
        if self.cap and self.cap.isOpened():
            self.running_camera = False
            self.cap.release()
            self.cap = None
            print("Camera đã tắt do chuyển chế độ")

    def run_recognition(self):
        mode = self.mode.get()
        if mode == "image":
            self.recognize_from_image()
        elif mode == "camera":
            self.recognize_from_camera()

    def recognize_from_image(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        img = Image.open(file_path)
        img_resized = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_resized)
        self.img_label.configure(image=img_tk)
        self.img_label.image = img_tk

        gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh.")
            return

        start_time = time.time()
        features = preprocess_and_extract_features(gray)
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000
        self.result_label.config(text=f"Cảm xúc dự đoán: {prediction}\n⏱️ Thời gian: {elapsed_time:.2f} ms")

    def recognize_from_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera.")
                return
            self.running_camera = True
            self.update_camera_frame()

    def update_camera_frame(self):
        if not self.running_camera or self.mode.get() != "camera":
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            messagebox.showerror("Lỗi", "Không thể đọc từ camera.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            start_time = time.time()
            features = preprocess_and_extract_features(gray)
            features_scaled = scaler.transform([features])
            features_pca = pca.transform(features_scaled)
            prediction = model.predict(features_pca)[0]
            elapsed_time = (time.time() - start_time) * 1000
        except Exception as e:
            prediction = "Không nhận diện được"
            elapsed_time = 0
            print("Lỗi xử lý:", e)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_resized = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_resized)
        self.img_label.configure(image=img_tk)
        self.img_label.image = img_tk

        self.result_label.config(text=f"Cảm xúc dự đoán: {prediction}\n⏱️ Thời gian: {elapsed_time:.2f} ms")

        self.root.after(100, self.update_camera_frame)

# ==== Khởi chạy ứng dụng ====
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognizerGUI(root)

    def on_closing():
        if app.cap and app.cap.isOpened():
            app.cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
