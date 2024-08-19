import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score

from models.cnn_model import create_cnn_model
from models.rf_model import create_rf_model

# 1. 데이터 로드 및 전처리
data = pd.read_csv('/Users/songle/Downloads/Alzhimer/data/oasis_longitudinal_demographics-8d83e569fa2e2d30.csv')  # CSV 파일 경로를 수정하세요
data['label'] = data['Group'].apply(lambda x: 1 if x == 'Demented' else 0)
X = data[['Age', 'EDUC', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. CNN 모델 학습 및 평가
# CNN 모델
X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1, 1)
X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1, 1)
cnn_model = create_cnn_model((X_train.shape[1], 1, 1))

# **여기에서 모델 컴파일을 추가합니다**
cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=16, validation_split=0.2)

# 3. 랜덤 포레스트 모델 학습 및 평가
rf_model = create_rf_model()
rf_model.fit(X_train, y_train)

# 4. 앙상블 예측 및 평가
cnn_predictions = cnn_model.predict(X_test_cnn).flatten()
rf_predictions = rf_model.predict(X_test)
combined_predictions = (cnn_predictions + rf_predictions) / 2
final_predictions = np.where(combined_predictions > 0.5, 1, 0)

final_accuracy = accuracy_score(y_test, final_predictions)
print(f'Ensemble Model Accuracy: {final_accuracy:.4f}')
