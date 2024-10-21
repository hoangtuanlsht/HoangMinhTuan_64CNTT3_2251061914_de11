import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#age: Tuổi
#sex: Giới tính
#cp: Đau ngực (kiểu đau ngực - chest pain type)
#trestbps: Huyết áp nghỉ (resting blood pressure)
#chol: Mức cholesterol
#fbs: Đường huyết đói (Fasting blood sugar)
#restecg: Điện tâm đồ lúc nghỉ (Resting electrocardiographic results)
#thalach: Nhịp tim tối đa đạt được (Maximum heart rate achieved)
#exang: Đau thắt ngực khi gắng sức (Exercise induced angina)
#oldpeak: Độ giảm ST sau gắng sức (ST depression induced by exercise relative to rest)
#slope: Độ dốc của đoạn ST
#ca: Số lượng mạch vành chính (Number of major vessels colored by fluoroscopy)
#thal: Tình trạng thalassemia
#target: Mục tiêu (thường biểu thị có bệnh hoặc không bệnh)

# 1. Load dữ liệu
df = pd.read_csv('heart.csv')

# 2. Chọn các đặc trưng (features) và nhãn mục tiêu (target)
X = df.drop('target', axis=1)  # Các đặc trưng
y = df['target']  # Nhãn mục tiêu

# 3.xử lý các dữ liệu thiếu
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
# 4. Tính Z-score và loại bỏ các outliers
z_scores = np.abs(stats.zscore(X))
filtered_entries = (z_scores < 3).all(axis=1)
X_filtered = X[filtered_entries]
y_filtered = y[filtered_entries]

# 5. Chuẩn hóa các biến độc lập
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
# 6. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)


# 7. Khởi tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=100)  # max_iter được đặt để đảm bảo hội tụ
model.fit(X_train, y_train)

# 8. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# 9. Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# 10. In kết quả
print("Confusion Matrix:")
print(conf_matrix)
print(f'Độ chính xác: {accuracy:.2f}')
print(f'F1-score: {f1:.2f}')
vi_du = pd.DataFrame({
    'age': [34,43],
    'sex': [0,0],
    'cp': [1,0],
    'trestbps': [118,132],
    'chol': [210,341],
    'fbs': [0,1],
    'restecg': [1,0],
    'thalach': [192,136],
    'exang': [0,1],
    'oldpeak': [0.7,3],
    'slope': [2,1],
    'ca': [0,0],
    'thal': [2,3]
})
vi_du_scaled = scaler.transform(vi_du)

# 2. Dự đoán trên dữ liệu ví dụ
y_pred_vi_du = model.predict(vi_du_scaled)

# 3. In kết quả dự đoán
print(f'Kết quả dự đoán: {y_pred_vi_du}')
"""f1_scores = []
max_iter_values = range(100, 2000, 100)  # Các giá trị max_iter từ 100 đến 2000, tăng dần 100

#  Vòng lặp để huấn luyện và tính F1-score cho từng giá trị max_iter
for iter_value in max_iter_values:
    model = LogisticRegression(max_iter=iter_value)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Tính F1-score và lưu lại
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

#  Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.plot(max_iter_values, f1_scores, marker='o', linestyle='-', color='b')
plt.title('Sự phụ thuộc của F1-score vào max_iter')
plt.xlabel('max_iter')
plt.ylabel('F1-score')
plt.grid(True)
plt.show()
"""

from sklearn.ensemble import RandomForestClassifier

# 1. Load dữ liệu
df = pd.read_csv('heart.csv')

# 2. Chọn các đặc trưng (features) và nhãn mục tiêu (target)
X = df.drop('target', axis=1)  # Các đặc trưng
y = df['target']  # Nhãn mục tiêu

# 3. Xử lý các dữ liệu thiếu
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 4. Tính Z-score và loại bỏ các outliers
z_scores = np.abs(stats.zscore(X))
filtered_entries = (z_scores < 3).all(axis=1)
X_filtered = X[filtered_entries]
y_filtered = y[filtered_entries]

# 5. Chuẩn hóa các biến độc lập
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# 6. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

# 7. Khởi tạo và huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators là số lượng cây trong rừng
model.fit(X_train, y_train)

# 8. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# 9. Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 10. In kết quả
print("Confusion Matrix:")
print(conf_matrix)
print(f'Độ chính xác: {accuracy:.2f}')
print(f'F1-score: {f1:.2f}')

vi_du = pd.DataFrame({
    'age': [34,43],
    'sex': [0,0],
    'cp': [1,0],
    'trestbps': [118,132],
    'chol': [210,341],
    'fbs': [0,1],
    'restecg': [1,0],
    'thalach': [192,136],
    'exang': [0,1],
    'oldpeak': [0.7,3],
    'slope': [2,1],
    'ca': [0,0],
    'thal': [2,3]
})
vi_du_scaled = scaler.transform(vi_du)

# 2. Dự đoán trên dữ liệu ví dụ
y_pred_vi_du = model.predict(vi_du_scaled)

# 3. In kết quả dự đoán
print(f'Kết quả dự đoán: {y_pred_vi_du}')