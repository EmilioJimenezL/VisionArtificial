import cv2
import numpy as np
import pytorchClassifier

from classicML import preprocess_image, load_model

cap = cv2.VideoCapture(7)
m_l_model = load_model()
n_n_model = pytorchClassifier.load_model("resnet18_cifar100_potentiated.pth")

while True:
    ret, frame = cap.read()
    resize_frame = cv2.resize(frame, (256,256))
    gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = preprocess_image(gray)
    resized_processed_frame = processed_frame.reshape(1, -1)
    pred = m_l_model.predict(resized_processed_frame)
    if pred[0] == 0:
        print("Bike not in frame")
    elif pred[0] == 1:
        print("Bike in frame")
        pred = pytorchClassifier.predict_frame(frame, n_n_model)
        print(pred)
    resized_frame = (processed_frame.reshape(gray.shape)*255).astype(np.uint8)
    cv2.imshow("frame", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()