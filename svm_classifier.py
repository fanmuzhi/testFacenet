import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle


data = np.load(r"./data/syna_embeddings.npz")
X_train, y_train, X_val, y_val = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
print('Dataset: train=%d, test=%d'  %  (X_train.shape[0],  y_train.shape[0]))
#norm X
in_encoder = Normalizer(norm="l2")
X_train = in_encoder.transform(X_train)
X_val = in_encoder.transform(X_val)

#norm Y
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)
y_val = out_encoder.transform(y_val)

#train
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

#predict
yhat_train = model.predict(X_train)
yhat_val = model.predict(X_val)
print(yhat_val)

#score
score_train = accuracy_score(y_train, yhat_train)
score_test = accuracy_score(y_val, yhat_val)

print("Accuracy: train=%.3f, test=%.3f" % (score_train*100, score_test*100))

pickle.dump(model, open("./data/syna_svc.model", 'wb'))

