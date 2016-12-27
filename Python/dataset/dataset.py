import os
import requests
import zipfile
import io
import cv2
import numpy as np

class HockeyDataset:
    def __init__(self, source_url, source_dir, max_size=None):
        self.url = source_url
        self.dir = source_dir
        self.max_size = max_size
    
    def read_names(self):
        if not os.path.exists(self.dir):              
            def get_content():                        
                response = requests.get(self.url, stream=True)
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
            
            zip_raw = get_content()
            zip_object = io.BytesIO(zip_raw)
            zip_file = zipfile.ZipFile(zip_object)
            zip_file.extractall(self.dir)
            zip_file.close()
    
        def get_avi():
            for name in os.listdir(self.dir):
                if name.endswith('.avi'):
                    yield name
        
        result = []
        if self.max_size:
            size = self.max_size // 2            
            ls = list(get_avi())
            result = ls[:size]
            result.extend(ls[-size:])
        else:
            result = list(get_avi())
        
        np.random.shuffle(result)
        return np.array(result)

    def split(self, x, frac = 1.0):
        indices = np.random.permutation(x.shape[0])
        train = (int)(x.shape[0] * frac)
        training_idx, test_idx = indices[:train], indices[train:]
        return x[training_idx], x[test_idx]    

    def read_avi(self, name):
        cap = cv2.VideoCapture(os.path.join(self.dir, name))
        result = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
        
            assert(frame.shape == (288, 360, 3))
            result.append(frame)
        cap.release()
        return np.array(result)/255.

    def read_tuple(self, name):
        y = 1 if name.startswith("fi") else 0
        x = self.read_avi(name)
        return (x, y)