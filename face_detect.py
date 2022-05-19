# function for face detection with mtcnn
import os
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
detector = MTCNN()

G_GREEN = (0, 255, 0)
G_RED = (255, 0, 0)
G_YELLOW = (255, 255, 0)
G_BLUE = (0, 0, 255)


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = None
    results = detector.detect_faces(image)
    if len(results) >= 1:
        # we expected the training image only contains 1 face
        x1, y1, width, height = results[0]['box']
        face = image[y1: y1 + height, x1:x1 + width]
        face = cv2.resize(face, required_size, interpolation=cv2.INTER_AREA)
        face = np.asarray(face)

        cv2.imshow("test", face)
        cv2.waitKey(0)

        # print("found face", filename)
    else:
        print("!!!found no face!!!", filename)

    return face


def test_live_cam():
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("camera", 1)

    while True:
        ret, inputImg = capture.read()
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(inputImg)

        if len(faces) >= 1:
            for i in range(len(faces)):
                x1, y1, width, height = faces[i]['box']
                x1 = abs(x1)
                y1 = abs(y1)
                x2 = x1 + width
                y2 = y1 + height
                face = inputImg[y1: y1 + height, x1:x1 + width]
                cv2.rectangle(inputImg, (x1, y1), (x2, y2), G_GREEN, 2)
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_RGB2BGR)
        cv2.imshow("camera", inputImg)
        key = cv2.waitKey(1)
        if key == 27:   # key = Esc
            break

    capture.release()  # release camera resource
    cv2.destroyAllWindows()
    print('Finished.')


# load the photo and extract the face
if __name__ == "__main__":
    # face = extract_face(
    #    r'C:\PythonProjects\testFacenet\test_image\syna\training\Henry\10.jpg')
    test_live_cam()
