from api import app
from datetime import datetime

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
session = {}

@app.route('/api/upload', methods=["POST"])
@ajax
def api_upload():
    fileName = '{0}{1}{2}'.format('zz', ''.join(random.sample(char_set, 8)), '.mp4')
    path = os.path.join('api', app.config['UPLOAD_FOLDER'], fileName)
    
    file = flask.request.files['videoFile']
    file.save(path)

    session_id = ''.join(random.sample(char_set, 10))
    now = datetime.now()
    session[session_id] = {'start': now, 'path': path}
    return {'session': session_id}

@app.route('/api/session/<id>')
@ajax
def api_session(id):
    session_info = session[id]
    elapsed = (datetime.now() - session_info['start']).total_seconds()
    if elapsed < 10:
        return {'video': None, 'time': None}
    
    elif elapsed < 30:
        return {'video': session_info['path'], 'time': None}

    stamps = [];
    for count in range(40):
        stamps.append({
                'fightStart': count % 2 == 0,
                'timeStamp': count * 10 
            })

    return {'video': session_info['path'], 'time': stamps}