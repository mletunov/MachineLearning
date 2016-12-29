import matplotlib.pyplot as plt
import dataset as ds
import learning

source_url = 'https://datastora.blob.core.windows.net/datasets/HockeyFights.zip'
#source_url = 'http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip'
#source_url = 'https://github.com/marmelroy/Zip/raw/master/ZipTests/bb8.zip'
source_dir = 'Data'

ds.run_tests()

# use only 50 videos to train and test network

dataset = ds.HockeyDataset(source_url, source_dir, max_size=100)

# full video network
# network = ln.FullVideoNetwork()
# graph = network.build(rnn_state=20)
# losses, accuracies = network.train(graph, dataset, 2)

# batch video network
#network = learning.BatchVideoNetwork(model_file=r"Models\batch\data")
network = learning.CnnVideoNetwork(model_file=r"Models\cnn\data")
graph = network.build(rnn_state=50, num_steps=30)
losses, accuracies = network.train(graph, dataset, epochs=5, batch_size=20)

plt.plot(losses)
plt.plot(accuracies)
plt.show()