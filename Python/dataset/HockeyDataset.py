import numpy as np
from PIL import Image
import dataset.utils as utils

class HockeyDataset:
    def __init__(self, source_url, source_dir, frame, max_size=None):
        self.url = source_url
        self.dir = source_dir
        self.frame = frame
        self.max_size = max_size

    def read_names(self, shuffle=False):
        if not utils.path_exists(self.dir):
            data = utils.download(self.url)
            utils.unzip(data, self.dir)

        result = list(utils.search_files(self.dir, ".avi"))

        if self.max_size and self.max_size < len(result):
            size = self.max_size // 2
            temp = result[:size]
            temp.extend(result[-(self.max_size - size):])
            result = temp

        if shuffle:
            np.random.shuffle(result)

        return result

    def read_avi(self, name):
        def frame_func(frame):
            img = Image.fromarray(frame)
            resized = img.resize((self.frame[1], self.frame[0]), Image.ANTIALIAS)
            result = np.asarray(resized)
            return result/255.

        return utils.read_avi(utils.path_join(self.dir, name), frame_func)

    def read_tuple(self, name):
        y = 1 if name.startswith("fi") else 0
        x = self.read_avi(name)
        return (x, y)

    # by_video - batch contains data from only one video
    # frames_count - max frames count in each epoch
    # batch_size - epochs count in one batch
    def gen_dataset(self, names=None, by_video=True, frames_count=None, batch_size=None):
        batch = []
        if names is None:
            names = self.read_names()

        def process(batch):
            x, y, names = zip(*batch)
            batch.clear()
            return np.array(x), np.array(y), np.array(names)

        for name in names:
            x, y = self.read_tuple(name)
            if not frames_count:
                batch.append((x, y, name))
            else:
                for shift in range(len(x) - frames_count + 1):
                    batch.append((x[shift:shift + frames_count], y, name))
                    if batch_size and len(batch) >= batch_size:
                        yield process(batch)
            if len(batch) > 0 and (by_video or not batch_size):
                yield process(batch)

        if len(batch) > 0:
            yield process(batch)