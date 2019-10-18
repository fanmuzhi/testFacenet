from mtcnn.mtcnn import MTCNN
import cv2

detector = MTCNN()


if __name__ == "__main__":
    #inputImg = cv2.imread("./test_image/lfw/George_W_Bush/George_W_Bush_0003.jpg")
    #inputImg = cv2.imread(r"E:/deeplearning/test_image/tom2.jpg")

    capture = cv2.VideoCapture(0)
    cv2.namedWindow("camera", 1)

    while True:
        ret, inputImg = capture.read()
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
        #x,y,z = inputImg.shape
        #x_scaled = (int)(x / 10)
        #y_scaled = (int)(y / 10)
        #inputImg = cv2.resize(inputImg, (y_scaled, x_scaled), interpolation=cv2.INTER_AREA)

        faces = detector.detect_faces(inputImg)
        #[{
        #  'box': [277, 90, 48, 63],
        #  'keypoints': {'nose': (303, 131), 'mouth_right': (313, 141), 'right_eye': (314, 114), 'left_eye': (291, 117), 'mouth_left': (296, 143)},
        #  'confidence': 0.99851983785629272
        #  }]
        if len(faces) >= 1:
            for i in range(len(faces)):
                x1, y1, width, height = faces[i]['box']
                x2 = x1 + width
                y2 = y1 + height
                cv2.rectangle(inputImg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(inputImg, "face" + str(i), (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_RGB2BGR)
        cv2.imshow("camera", inputImg)
        key = cv2.waitKey(1)
        if key == 27:   # key = Esc
            break

    cv2.destroyAllWindows()

