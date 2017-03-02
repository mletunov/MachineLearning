"""
The flask application package.
"""
import os
from flask import Flask

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Temp'
os.makedirs(os.path.join('web', app.config['UPLOAD_FOLDER']), exist_ok=True)

import web.views
