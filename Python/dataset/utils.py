import numpy as np

def download(url):    
    import requests

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception("Cannot download data, code={0}".format(response.status_code))

    total_length = int(response.headers.get('content-length'))
    length = 0
    out = bytearray()
    for data in response.iter_content(chunk_size=4096):
        length += len(data)
        out += data
        done = (50 * length) // total_length
        percents = (100 * length) // total_length
        print("\r", "Downloading [{0}{1}] {2}%".format('=' * done, ' ' * (50-done), percents), end="")
    print("\r", "Download completed")
    return out

def unzip(raw_data, dir):
    import io
    import zipfile
    
    zip_object = io.BytesIO(raw_data)
    zip_file = zipfile.ZipFile(zip_object)
    zip_file.extractall(dir)
    zip_file.close()

def path_exists(path):
    import os
    return os.path.exists(path)

def search_files(dir, ends=None):
    import os

    for name in os.listdir(dir):
        if not ends or name.endswith(ends):
            yield name

def read_avi(path, frame_func=None):
    import cv2
    
    cap = cv2.VideoCapture(os.path.join(self.dir, name))
    result = []
    frame_func = frame_func or (lambda frame: frame)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
                        
        result.append(frame_func(frame))
    cap.release()
    return np.array(result)

def split(self, array, frac):
    temp = np.array(array)    
    indices = np.random.permutation(temp.shape[0])
    part = (int)(temp.shape[0] * frac)
    part1_idx, part2_idx = indices[:part], indices[part:]
    return temp[part1_idx], temp[part2_idx]