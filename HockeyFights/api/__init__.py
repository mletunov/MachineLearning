"""
The flask application package.
"""
import os
from flask import Flask
from flask_cors import CORS


app = Flask(__name__)

cors = CORS(app)
app.config['UPLOAD_FOLDER'] = 'Temp'
app.config['CORS_HEADERS'] = 'Content-Type'
os.makedirs(os.path.join('api', app.config['UPLOAD_FOLDER']), exist_ok=True)

import api.routes