from api import app
from api import tasks
from flask_cors import cross_origin
from flask import send_from_directory
import flask, os


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

session = {}

@app.route('/api/upload', methods=["POST"])
@cross_origin()
@ajax
def api_upload():
    file = flask.request.files['videoFile']
    ext = os.path.splitext(file.filename)[-1].lower()
    
    fileName = '{0}{1}{2}'.format('zz', tasks.generate_name(8), ext) 
    path = os.path.join(os.path.join('api', app.config['UPLOAD_FOLDER']), fileName)
    file.save(path)

    session_id = tasks.generate_name(5)
    session[session_id] = {'fileName': fileName}
    return {'session': session_id}   

@app.route('/api/session/<id>')
@cross_origin()
@ajax
def api_session(id):
    session_info = session[id]
    if 'path' not in session_info:
        tasks.save_mp4(session_info, os.path.join('api', app.config['UPLOAD_FOLDER']))
        return {'video': session_info['path'], 'time': None}
    
    elif 'time' not in session_info:
        tasks.calc_times(session_info)

    return {'video': session_info['path'], 'time': session_info['time']}

@app.route('/video/<path>')
@cross_origin()
def send_js(path):
    return send_from_directory(app.config['UPLOAD_FOLDER'], path)