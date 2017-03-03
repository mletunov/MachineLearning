from learning.rnn import *
from learning import baseNetwork
import dataset as ds

def create_model(type, frame, state, steps, norm, dir=None):
    def full_model(path):
        return lambda: fullVideo.FullModel(
            frame=frame, norm_type=norm, checkpoint_dir=path,
            seed=100).build(rnn_state=state, avg_result=True)

    def simple_model(path):
        return lambda: simpleVideo.SimpleModel(
            frame=frame, norm_type=norm, checkpoint_dir=path,
            seed=100).build(rnn_state=state, num_steps=steps, avg_result=True)

    def cnn_model(path, batch_norm, dropout):
        return lambda: cnnVideo.CnnModel(
            frame=frame, norm_type=norm, checkpoint_dir=path,
            seed=100).build(rnn_state=state, num_steps=steps, avg_result=True, batch_norm=batch_norm, dropout=dropout)

    folder = "Model"
    models = {
        "RNN_FULL": full_model(ds.utils.path_join(folder, dir or "full")),
        "RNN_SIMPLE": simple_model(ds.utils.path_join(folder, dir or "simple")),
        "RNN_CNN": cnn_model(ds.utils.path_join(folder, dir or "cnn"), False, False),
        "RNN_CNN_BATCH":  cnn_model(ds.utils.path_join(folder, dir or "batch_norm"), True, False),
        "RNN_CNN_DROP": cnn_model(ds.utils.path_join(folder, dir or "drop"), False, True),
        "RNN_CNN_BATCH_DROP": cnn_model(ds.utils.path_join(folder, dir or "batch_drop"), True, True),
    }

    return models[type]()


def create_trainer(type, model, rate):
    def cnn_trainer():
        return lambda: cnnVideo.CnnTrainer(model).build(learning_rate=rate)

    trainers = {
        "RNN_FULL": lambda: fullVideo.FullTrainer(model).build(learning_rate=rate),
        "RNN_SIMPLE": lambda: simpleVideo.SimpleTrainer(model).build(learning_rate=rate),
        "RNN_CNN": cnn_trainer(),
        "RNN_CNN_BATCH": cnn_trainer(),
        "RNN_CNN_DROP": cnn_trainer(),
        "RNN_CNN_BATCH_DROP": cnn_trainer(),
    }

    return trainers[type]()


def create_predictor(type, model):
    return baseNetwork.Predictor(model)

def local_dataset(frame, max_size=None):
    source_url = 'https://datastora.blob.core.windows.net/datasets/HockeyFights.zip'
    # source_url = 'http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip'
    source_dir = 'Data'
    return ds.HockeyDataset(source_url, source_dir, frame, max_size=max_size)

#def video_dataset(fileName):
    