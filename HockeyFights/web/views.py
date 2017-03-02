"""
Routes and views for the flask application.
"""

from datetime import datetime
from web import app

import tempfile
import flask
import os

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
        fileName = tempfile.mkstemp('.avi', 'zz', app.config['UPLOAD_FOLDER'])[1]
        url = app.config['UPLOAD_FOLDER'] + '/' + os.path.basename(fileName)
        with open(fileName, 'wb') as file:
            chunk_size = 4096
            while True:
                chunk = flask.request.stream.read(chunk_size)
                if len(chunk) == 0:
                    break

                file.write(chunk)
        return flask.jsonify({'success': True , 'fileName': url})
    except Exception as ex:
        return flask.jsonify({'success': False, 'message': ex})