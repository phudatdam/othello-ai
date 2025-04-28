import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Đọc và tiền xử lý dữ liệu
df = pd.read_csv('othello_features.csv')

weights_per_phase = []
scalers = []  # Lưu scaler cho từng phase

for phase in range(6):
    phase_data = df[df['phase'] == phase]
    # Kiểm tra đủ dữ liệu
    if len(phase_data) < 50:
        print(f"Phase {phase} skipped (only {len(phase_data)} samples)")
        weights_per_phase.append(np.zeros(4))
        scalers.append(None)
        continue
        
    # Chuẩn bị dữ liệu
    X = phase_data[['corner_diff', 'mobility', 'frontier_discs', 'stability']]
    y = phase_data['score_diff']
    
    # Chuẩn hóa
    scalers.append(None)
    
    # Chia tập train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
    )
    
    # Huấn luyện
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nPhase {phase}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print("-------------------")
    # Lưu weights
    weights_per_phase.append(model.coef_)

print(np.array(weights_per_phase))
# Tính correlation matrix
corr_matrix = df[['corner_diff', 'mobility', 'frontier_discs', 'stability', 'score_diff']].corr()
print(corr_matrix['score_diff'].sort_values(ascending=False))

# Lưu weights và scalers
np.save('linear_phase_weights.npy', np.array(weights_per_phase))