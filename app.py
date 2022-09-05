from flask import Flask,render_template, request, make_response, Response
from tensorflow.keras.models import load_model
from functions import *
from functools import wraps
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
import os
from cv2 import VideoCapture
from cv2 import waitKey
from PIL import Image
import cv2


model = load_model('model.h5')

app = Flask(__name__)

picfolder= os.path.join('static','images')

app.config['UPLOAD_FOLDER'] = picfolder

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if auth and auth.username == 'username4' and auth.password == 'passsword':
            return f(*args, **kwargs)

        return make_response('Could not verify!',401, {'WWW-Authenticate' : 'Basic realm="Login Required"'})
    return decorated



@app.route('/', methods=['GET'])
def home():
    if request.authorization and request.authorization.username == 'username5' and request.authorization.password == 'password':
        return render_template('index.html')

    return make_response('Could not verify!',401, {'WWW-Authenticate' : 'Basic realm="Login Required"'})

@app.route('/',methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path= 'images/' + imagefile.filename
    imagefile.save(image_path)
    x, df = ocr_reading(image_path)
    return render_template('index.html' , x = x, tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/predict', methods=['POST', 'GET'])
def predict2():
    x, df = ocr_reading('test1.jpg')
    return render_template('index.html' , x = x, tables=[df.to_html(classes='data')], titles=df.columns.values)


@app.route('/deck')
def Deck():
    data = pd.read_csv('userdata.csv')
    cm = data['cardmarket'].sum().round()
    tcg = data['tcgplayer'].sum().round()
    eb = data['ebay'].sum().round()
    az = data['amazon'].sum().round()
    cs = data['coolstuffinc'].sum().round()
    return render_template('deck.html', tables=[data.to_html()], titles=[''], cm = cm, tcg = tcg, eb = eb, az = az, cs = cs)

@app.route('/video_feed')
def video_feed():
    return Response(open_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/open_camera')
def open_camera():
    camera = cv2.VideoCapture(0)
    while True:
        _, image = camera.read()
        cv2.imshow('Text detection', image)
        cv2.setWindowProperty('Text detection', cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF==ord('s'):
            cv2.imwrite('test1.jpg', image)
            break
    camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    app.run(port=4001,debug=True)