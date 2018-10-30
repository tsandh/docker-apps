#! /usr/bin/env/python
import flask
from flask import send_from_directory

app = flask.Flask(__name__)
import os
import json
import sh
import face_recognition
import pickle


def save_faces():
  f = open("model/model.pkl",'wb')
  pickle.dump(known_faces,f) 
  f.close()
  f = open("model/index.json",'w')
  f.write(json.dumps(face_index))
  f.close()

@app.route('/train', methods=['POST'])
def train_model():  # pylint: disable=unused-variable
    tag = flask.request.args.get('tag')
    image_file = flask.request.files['image']
    image_file.save("train_image.jpg")
    train_image = face_recognition.load_image_file("train_image.jpg")
    train_face_encoding = face_recognition.face_encodings(train_image)[0]

    known_faces.append(train_face_encoding)
    face_index.append(tag)
    save_faces()

    result = json.dumps({"tag": tag})
    return flask.Response(status=200, response=result + "\n", mimetype='application/json')

@app.route('/predict', methods=['POST'])
def predict_model():  # pylint: disable=unused-variable
    image_file = flask.request.files['image']
    image_file.save("predict_image.jpg")

    predict_image = face_recognition.load_image_file("predict_image.jpg")
    predict_face_encoding = face_recognition.face_encodings(predict_image)[0]

    results = face_recognition.compare_faces(known_faces, predict_face_encoding)

    predicted = -1
    for i in range(0,len(results)):
        if results[i]:
            predicted = i
            break
    tag = ""
    if predicted != -1:
        tag = face_index[predicted]

    result = json.dumps({"prediction": tag})
    return flask.Response(status=200, response=result + "\n", mimetype='application/json')

try:
  f = open("model/model.pkl",'rb')
  known_faces = pickle.load(f) 
  f.close()
  f = open("model/index.json")
  face_index = json.loads(f.read())
  f.close()
except:
  known_faces = []
  face_index = []

app.run("0.0.0.0", port=5000)
