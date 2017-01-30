import matplotlib.pyplot as plt
import dataset as ds
import learning

ds.run_tests()


source_url = 'https://datastora.blob.core.windows.net/datasets/HockeyFights.zip'
#source_url = 'http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip'
#source_url = 'https://github.com/marmelroy/Zip/raw/master/ZipTests/bb8.zip'
source_dir = 'Data'
dataset = ds.HockeyDataset(source_url, source_dir, max_size=1000)

class Mode:
    TRAIN = 0,
    PREDICT = 1

class Model:
    RNN_CNN_AVG = 0
    RNN_CNN_BATCH = 1
    RNN_CNN_DROP = 2
    RNN_CNN_BATCH_DROP = 3

mode = Mode.TRAIN
type = Model.RNN_CNN_DROP

if type == Model.RNN_CNN_AVG:
    model = learning.cnnVideo.CnnModel(checkpoint_dir='Models/cnn_50', seed=100).build(rnn_state=50, num_steps=30, avg_result=True)
    if mode == Mode.TRAIN:
        trainer = learning.cnnVideo.CnnTrainer(model).build(learning_rate=1e-2)
        losses, accuracies = trainer.train(dataset, epochs=5, batch_size=20)

if type == Model.RNN_CNN_BATCH:
    model = learning.cnnVideo.CnnModel(checkpoint_dir='Models/batch_norm', seed=100).build(rnn_state=50, num_steps=30, avg_result=True, batch_norm=True)
    if mode == Mode.TRAIN:
        trainer = learning.cnnVideo.CnnTrainer(model).build(learning_rate=1e-2)
        losses, accuracies = trainer.train(dataset, epochs=5, batch_size=20)

if type == Model.RNN_CNN_DROP:
    model = learning.cnnVideo.CnnModel(checkpoint_dir='Models/drop', seed=100).build(rnn_state=50, num_steps=30, avg_result=True, dropout=True)
    if mode == Mode.TRAIN:
        trainer = learning.cnnVideo.CnnTrainer(model).build(learning_rate=1e-2)
        losses, accuracies = trainer.train(dataset, epochs=5, batch_size=20)

if type == Model.RNN_CNN_BATCH_DROP:
    model = learning.cnnVideo.CnnModel(checkpoint_dir='Models/batch_drop', seed=100).build(rnn_state=50, num_steps=30, avg_result=True, batch_norm=True, dropout=True)
    if mode == Mode.TRAIN:
        trainer = learning.cnnVideo.CnnTrainer(model).build(learning_rate=1e-2)
        losses, accuracies = trainer.train(dataset, epochs=5, batch_size=20)

if mode == Mode.PREDICT and model:
    predictor = learning.Predictor(model)
    result = predictor.predict(dataset, batch_size=20)
    print("Accuracy:", predictor.accuracy(result))