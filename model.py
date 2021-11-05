# Now only simple regression
# just used to test the dataloader and the metrics
import time

import numpy as np
from sklearn.linear_model import Lasso

from dataloader import load_dynamic_data
from metrics import compute_metric


tick = time.time()
X_train, y_train_arousal, y_train_valence, X_test, y_test_arousal, y_test_valence, music_id_train, music_id_test = load_dynamic_data()
print("Data loaded!", time.time() - tick)

model_arousal = Lasso()
model_arousal.fit(X_train, y_train_arousal)
y_train_arousal_pred = model_arousal.predict(X_train)
y_test_arousal_pred = model_arousal.predict(X_test)

model_valence = Lasso()
model_valence.fit(X_train, y_train_valence)
y_train_valence_pred = model_arousal.predict(X_train)
y_test_valence_pred = model_arousal.predict(X_test)


results_train = compute_metric((y_train_arousal_pred, y_train_valence_pred), (y_train_arousal, y_train_valence), music_id_train)
results_test = compute_metric((y_test_arousal_pred, y_test_valence_pred), (y_test_arousal, y_test_valence), music_id_test)
print("Training")
print(results_train)
print("Test")
print(results_test)