import cv2 as cv


def preprocessImage(image_path, size=(128, 128)):
    original = cv.imread(image_path)
    img = cv.resize(original, size)
    cv.imshow("Preprocessed Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return img
