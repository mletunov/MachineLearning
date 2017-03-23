from api import app

import flask, os
import random
import string
import learning

def ajax(func):
    def wrapper(*args, **kwargs):
        try:
            dict = func(*args, **kwargs) or {}
            return flask.jsonify(dict)
        except Exception as ex:
            dict = {'exception' : str(ex)}
            return flask.make_response(flask.jsonify(dict), 500)

    wrapper.__name__ = func.__name__
    return wrapper

char_set = string.ascii_lowercase + string.digits

@app.route('/upload', methods=["POST"])
@ajax
def upload():
    fileName = '{0}{1}{2}'.format('zz', ''.join(random.sample(char_set, 8)), '.mp4')
    path = os.path.join('api', app.config['UPLOAD_FOLDER'], fileName)
    url = '/video/{0}'.format(fileName)

    file = flask.request.files['videoFile']
    file.save(path)
    return {'filePath': url, 'fileName': fileName}

@app.route('/video/<fileName>')
def download_file(fileName):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'], fileName, as_attachment=True)

@app.route('/time/<path:fileName>')
@ajax
def get_time_stamps(fileName):
    stamps = [];
    for count in range(40):
        stamps.append({
                'fightStart': count % 2 == 0,
                'timeStamp': count * 10 
            })

    return stamps

@app.route('/test')
@ajax
def test():
    path = os.path.join('web', app.config['UPLOAD_FOLDER'], flask.request.args.get('fileName'))
    import cv2
    
    predictor = app.config['predictor']
    
    learning.factory.video_dataset(path)
    cap = cv2.VideoCapture(path)
    ret = cap.isOpened();
