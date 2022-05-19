from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np

detector = MTCNN()

def training_extract_face(filename, required_size=(160, 160)):
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

        #cv2.imshow("test", face)
        # cv2.waitKey(0)

        #print("found face", filename)
    else:
        print("!!!found no face!!!", filename)

    return face


def training_load_faces(directory):
    faces = []
    labels = []

    if not os.path.exists(directory):
        print("direcotry doesn't exsist")

    for subdir in os.listdir(directory):
        sub_directory = directory + subdir + "/"
        if not os.path.isdir(sub_directory):
            continue

        filenum = 0
        for file in os.listdir(sub_directory):
            file = sub_directory + file
            if not os.path.isfile(file):
                continue
            face = training_extract_face(file)
            if face is not None:
                filenum += 1
                faces.append(face)
                labels.append(subdir)
        print("load %d samples for %s" % (filenum, subdir))
        filenum = 0

    faces = np.asarray(faces)
    labels = np.asarray(labels)
    return faces, labels


if __name__ == "__main__":
    X_train, y_train = training_load_faces("./test_image/syna/training/")
    X_val, y_val = training_load_faces("./test_image/syna/validate/")

    print("training data size: ", X_train.shape)
    print("validate data size: ", X_val.shape)
    print("label:", y_train)

    np.savez_compressed("data/syna.npz", X_train, y_train, X_val, y_val)
