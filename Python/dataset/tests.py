import numpy as np
from dataset.HockeyDataset import HockeyDataset

class TestHockeyDataset(HockeyDataset):
    def __init__(self):
        self.names = []
        self.read = lambda name: []
    def read_names(self):
        return self.names
    def read_avi(self, name):
        return self.read(name)

dataset = TestHockeyDataset()
dataset.names = ["fi", "no"]
a1 = ([1, 2, 3, 4], 1)
a2 = ([5, 6, 7], 0)
def read(name):
    if name == "fi":
        return a1[0]
    if name == "no":
        return a2[0]
    return []
dataset.read = read

def test_by_video():
    result = list(dataset.gen_dataset(by_video=True, frames_count=None, batch_size=None))

    assert(len(result) == 2)

    result_x, result_y = result[0]
    assert(np.array_equal(result_x, np.array([a1[0]])))
    assert(np.array_equal(result_y, np.array([a1[1]])))

    result_x, result_y = result[1]
    assert(np.array_equal(result_x, np.array([a2[0]])))
    assert(np.array_equal(result_y, np.array([a2[1]])))

def test_frames_count(by_video):
    result = list(dataset.gen_dataset(by_video=by_video, frames_count=3, batch_size=None))

    assert(len(result) == 2)

    result_x, result_y = result[0]
    assert(np.array_equal(result_x, np.array([a1[0][0:3], a1[0][1:4]])))
    assert(np.array_equal(result_y, np.array([a1[1], a1[1]])))
        
    result_x, result_y = result[1]
    assert(np.array_equal(result_x, np.array([a2[0]])))
    assert(np.array_equal(result_y, np.array([a2[1]])))

def test_batch_size(by_video):
    result = list(dataset.gen_dataset(by_video=by_video, frames_count=2, batch_size=2))

    assert(len(result) == 3)

    result_x, result_y = result[0]
    assert(np.array_equal(result_x, np.array([a1[0][0:2], a1[0][1:3]])))
    assert(np.array_equal(result_y, np.array([a1[1], a1[1]])))
            
    if by_video:
        result_x, result_y = result[1]
        assert(np.array_equal(result_x, np.array([a1[0][2:4]])))
        assert(np.array_equal(result_y, np.array([a1[1]])))
            
        result_x, result_y = result[2]
        assert(np.array_equal(result_x, np.array([a2[0][0:2], a2[0][1:3]])))
        assert(np.array_equal(result_y, np.array([a2[1], a2[1]])))

    else:
        result_x, result_y = result[1]
        assert(np.array_equal(result_x, np.array([a1[0][2:4], a2[0][0:2]])))
        assert(np.array_equal(result_y, np.array([a1[1], a2[1]])))
            
        result_x, result_y = result[2]
        assert(np.array_equal(result_x, np.array([a2[0][1:3]])))
        assert(np.array_equal(result_y, np.array([a2[1]])))

def run():
    test_by_video()
    test_frames_count(by_video=True)
    test_frames_count(by_video=False)
    test_batch_size(by_video=True)
    test_batch_size(by_video=False)