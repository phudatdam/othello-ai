import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Đọc và tiền xử lý dữ liệu
df = pd.read_csv('othello_features.csv')
df['label'] = 100*(df['label'] == 1).astype(int)  # Chuyển label về 0/1

weights_per_phase = []
scalers = []  # Lưu scaler cho từng phase

for phase in range(5,65):
    phase_data = df[df['phase'] == phase]
    
    # Kiểm tra đủ dữ liệu
    if len(phase_data) < 50:
        print(f"Phase {phase} skipped (only {len(phase_data)} samples)")
        weights_per_phase.append(np.zeros(5))
        continue
        
    # Chuẩn bị dữ liệu
    X = phase_data[['score_diff', 'corner_diff', 'mobility', 'frontier_discs', 'stability']]
    y = phase_data['label']
    
    scalers.append(None)
    
    # Chia tập train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Huấn luyện
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Phase {phase}")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-------------------")
    
    # Lưu weights
    weights_per_phase.append(model.coef_[0])
rounded_weights = [np.round(w, 3) for w in weights_per_phase]
print(rounded_weights)
# Lưu weights và scalers
np.save('phase_weights.npy', np.array(rounded_weights))
np.save('phase_scalers.npy', np.array(scalers))