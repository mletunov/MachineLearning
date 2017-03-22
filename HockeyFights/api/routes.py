from api import app

import flask, os
from flask_cors import CORS, cross_origin

import random
import string
import learning

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

char_set = string.ascii_lowercase + string.digits

@app.route('/upload', methods=["POST"])
def upload():
    try:
        fileName = '{0}{1}{2}'.format('zz', ''.join(random.sample(char_set, 8)), '.mp4')
        path = os.path.join('web', app.config['UPLOAD_FOLDER'], fileName)
        url = '/video/{0}'.format(fileName)
        
        file = request.files['videoFile']

        if file:
            file.save(path)           
            return flask.jsonify({'success': True , 'filePath': url, 'fileName': fileName})
    except Exception as ex:
        return flask.jsonify({'success': False, 'message': ex})

@app.route('/video/<path:filename>')
def download_file(filename):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)

@app.route('/time/<path:filename>')
def get_time_stamps(filename):
    try:
        stamps = list();
        for count in range(40):
            stamps.append({
                   'fightStart': count % 2 == 0,
                   'timeStamp': count * 10 
                })
        return flask.jsonify(stamps);
    except Exception as ex:
        return flask.jsonify({'success': False, 'message': ex})

@app.route('/test')
def test():
    path = os.path.join('web', app.config['UPLOAD_FOLDER'], flask.request.args.get('fileName'))
    import cv2
    
    predictor = app.config['predictor']
    
    #learning.factory.video_dataset(path)
    try:
        cap = cv2.VideoCapture(path)
        ret = cap.isOpened();
        return flask.jsonify({'success': True})
    except Exception as ex:
        return flask.jsonify({'success': False, 'message': ex})