import utils
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

img = utils.preprocessImage("testing_images/testing4.jpeg")
cv.imshow("Image", img.reshape(128, 128))
cv.waitKey(0)
cv.destroyAllWindows()

data_dir = "data"  # base folder - Changed to point to the downloaded dataset
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(
    '.')]  # Filter out non-directory files starting with '.'
print("Clases detectadas:", classes)

X = []
y = []

for label, cls in enumerate(classes):
    counter = 0
    folder = os.path.join(data_dir, cls)
    print(f"Processing class: {folder}")
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            X.append(utils.preprocessImage(path))
            y.append(int(label))
            counter += 1
        except Exception as e:  # Catch specific exception for better debugging
            print(f"Skipping file: {path} - Error: {e}")
    print(f"Processed {counter} images in class {cls}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")

train_data = cv.ml.TrainData.create(X_train.astype(np.float32), cv.ml.ROW_SAMPLE, y_train.astype(np.float32))

finalModel = cv.ml.LogisticRegression.create()
finalModel.setLearningRate(0.05)
finalModel.setIterations(1000)
finalModel.setRegularization(cv.ml.LOGISTIC_REGRESSION_REG_L2)
finalModel.setTrainMethod(cv.ml.LOGISTIC_REGRESSION_BATCH)
finalModel.setMiniBatchSize(1)
finalModel.train(train_data)

_, predictions = finalModel.predict(X_test.astype(np.float32))
acc = accuracy_score(y_test, predictions)
matrix = confusion_matrix(y_test, predictions)
print(f"Accuracy: {acc * 100:.2f}%")
print("Confussion matrix:\n", matrix)

orig_path = "testing_images/testing6.jpeg"
original_img = cv.imread(orig_path)

pre_img = utils.preprocessImage(orig_path)
pre_img_float = pre_img.astype(np.float32).reshape(1, -1)
_, testPrediction = finalModel.predict(pre_img_float)

cv.imshow("Original", original_img)
cv.imshow("Preprocessed", pre_img.reshape(128, 128))
cv.setWindowTitle("Preprocessed", f"Preprocessed - Prediction: {int(testPrediction.ravel()[0])}")
cv.waitKey(0)
