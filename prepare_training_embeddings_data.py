import tensorflow.keras as keras
import numpy as np

def get_embedding(model, face):
    face = face.astype("float32")
    mean, std = face.mean(), face.std()
    face = (face - mean) / std

    samples = np.expand_dims(face, axis=0)
    yhat = model.predict(samples)

    return yhat[0]


if __name__ == "__main__":

    model = keras.models.load_model("./model/facenet_keras.h5")
    print(model.summary())

    data = np.load("./data/syna.npz")
    X_train, y_train, X_val, y_val = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
    print(X_train.shape, y_train.shape)

    newX_train = []
    for face in X_train:
        embedding = get_embedding(model, face)
        newX_train.append(embedding)
    newX_train = np.asarray(newX_train)

    newX_val = []
    for face in X_val:
        embedding = get_embedding(model, face)
        newX_val.append(embedding)
    newX_val = np.asarray(newX_val)

    print(newX_train.shape)
    print(newX_val.shape)

    np.savez_compressed("data/syna_embeddings.npz", newX_train, y_train, newX_val, y_val)

