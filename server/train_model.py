import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu đặc trưng đã tạo từ CSV
df = pd.read_csv("othello_features.csv")

# Đặc trưng đầu vào (X) và nhãn đầu ra (y)
X = df[["disc_diff", "corner_diff"]]
y = df["label"]

# Chia dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Mean squared error (MSE):", mean_squared_error(y_test, y_pred))
print("R^2 score:", r2_score(y_test, y_pred))

# In trọng số (weights)
print("\nModel coefficients (weights):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print("Intercept:", model.intercept_)