import os
import random
from ffmpy import FFmpeg
import string, random

char_set = string.ascii_lowercase + string.digits

def save_mp4(session_info, base_path):
    fileName = session_info['fileName']
    ext = os.path.splitext(fileName)[-1].lower()
    
    if ext != ".mp4":
        mp4_fileName = '{0}{1}{2}'.format('zz', generate_name(8), '.mp4')
        path = os.path.join(base_path, fileName)
        mp4_path = os.path.join(base_path, mp4_fileName)

        ff_convert = FFmpeg(inputs={path: None},
                            outputs={mp4_path: None})
        ff_convert.run()

        os.remove(path)
        fileName = mp4_fileName

    session_info['path'] = 'video/' + fileName

def calc_times(session_info):
    stamps = []
    for count in range(40):
        stamps.append({
                'fightStart': count % 2 == 0,
                'timeStamp': count * 10  
            })
    session_info['time'] = stamps

def generate_name(size):
    return ''.join(random.sample(char_set, size))