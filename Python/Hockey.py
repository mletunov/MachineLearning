import matplotlib.pyplot as plt
import dataset as ds
import learning

ds.run_tests()


source_url = 'https://datastora.blob.core.windows.net/datasets/HockeyFights.zip'
#source_url = 'http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip'
#source_url = 'https://github.com/marmelroy/Zip/raw/master/ZipTests/bb8.zip'
source_dir = 'Data'
dataset = ds.HockeyDataset(source_url, source_dir, max_size=500)

class Mode:
    RNN_SIMPLE = 0,
    RNN_CNN = 1,
    RNN_FULL = 2

mode = Mode.RNN_CNN

if mode == Mode.RNN_CNN:    
    model = learning.cnnVideo.CnnModel(checkpoint_dir='Models/cnn', seed=100).build(rnn_state=100, num_steps=30, avg_result=True)
    trainer = learning.cnnVideo.CnnTrainer(model).build(learning_rate=1e-2)
    losses, accuracies = trainer.train(dataset, epochs=5, batch_size=20)

if mode == Mode.RNN_SIMPLE:
    model = learning.batchVideo.BatchModel(checkpoint_dir='Models/batch', seed=100).build(rnn_state=50, num_steps=30, avg_result=True)
    trainer = learning.batchVideo.BatchTrainer(model).build(learning_rate=1e-2)
    losses, accuracies = trainer.train(dataset, epochs=5, batch_size=20)

if mode == Mode.RNN_FULL:
    network = learning.FullVideoNetwork()
    graph = network.build(rnn_state=20)
    losses, accuracies = network.train(graph, dataset, 2)

plt.plot(losses)
plt.plot(accuracies)
plt.show()