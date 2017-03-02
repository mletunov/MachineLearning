"""
Routes and views for the flask application.
"""

from datetime import datetime
from web import app

import flask
import os

import random
import string

char_set = string.ascii_lowercase + string.digits

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return flask.render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/upload', methods=["POST"])
def upload():
    try:
        fileName = '{0}{1}{2}'.format('zz', ''.join(random.sample(char_set, 8)), '.avi')
        path = os.path.join('web', app.config['UPLOAD_FOLDER'], fileName)
        url = '/video/{0}'.format(fileName)
        
        with open(path, 'wb') as file:
            chunk_size = 4096
            while True:
                chunk = flask.request.stream.read(chunk_size)
                if len(chunk) == 0:
                    break

                file.write(chunk)
            file.close()
        return flask.jsonify({'success': True , 'fileName': url})
    except Exception as ex:
        return flask.jsonify({'success': False, 'message': ex})

@app.route('/video/<path:filename>')
def download_file(filename):
    return flask.send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)