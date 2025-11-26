from pathlib import Path

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def open_image(filepath, size=(256,256), cmap='gray', verbose=False):
    if verbose: print('Opening image: {}'.format(filepath))
    if cmap == 'gray':
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(img, size)
    elif cmap == 'bgr':
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        return cv2.resize(img, size)
    else: raise ValueError('Invalid color map: {}'.format(cmap))

def open_images_with_labels(folder, label):
    folderPath = Path(folder)
    images, labels = [], []
    for file in folderPath.glob('*.jpeg'):
        images.append(open_image(file))
        labels.append(label)
    return images, labels

def preprocess_image(image):
    equalized = cv2.equalizeHist(image)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    normalized = canny / 255.0
    flattened = normalized.flatten()
    return flattened

def preprocess_images(image_list):
    processedImages = []
    for img in image_list:
        img = preprocess_image(img)
        processedImages.append(img)
    return processedImages

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    joblib.dump(model, 'model_zoo/svc/SVCModel.pkl')

def load_model():
    return joblib.load('model_zoo/svc/SVCModel.pkl')

if __name__ == '__main__':
    images0, labels0 = open_images_with_labels('data/classicMLTraining/0', 0)
    images1, labels1 = open_images_with_labels('data/classicMLTraining/1', 1)
    X0 = preprocess_images(images0)
    X1 = preprocess_images(images1)

    X = np.array(X0 + X1)
    y = np.array(labels0 + labels1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = load_model()
    accuracy = model.score(X_test, y_test)
    confMatrix = confusion_matrix(y_test, model.predict(X_test))

    print("model accuracy: ", accuracy)
    print("confusion matrix: \n", confMatrix)

    image = preprocess_image(open_image('data/classicMLTraining/1/12.jpeg'))
    imageReshaped = image.reshape(1, -1)
    prediction = model.predict(imageReshaped)

    plt.imshow(image.reshape(256, 256), cmap='gray')
    plt.title('Prediction: {}'.format(prediction[0]))
    plt.show()