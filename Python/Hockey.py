from batchVideo import network as batchNetwork
from fullVideo import network as fullNetwork
from cnnrnnVideo import network as cnnrnnNetwork
from dataset import dataset as ds
import numpy as np

source_url = 'https://datastora.blob.core.windows.net/datasets/HockeyFights.zip'
#source_url = 'http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip'
source_dir = 'Data'

# use only 50 videos to train and test network
np.random.seed(100)
dataset = ds.HockeyDataset(source_url, source_dir, max_size=50)
network = cnnrnnNetwork.build(num_steps=30)
cnnrnnNetwork.train(network, dataset, 4, batch_size=10)

#network = build_network2()
#train2(network, epochs=10)