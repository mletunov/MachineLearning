import dataset as ds
import learning
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Network functional mode", required=True, choices=["TRAIN", "PREDICT"])
    parser.add_argument("-t", "--type", help="Network type", required=True,
                        choices=["RNN_FULL", "RNN_SIMPLE", "RNN_CNN", "RNN_CNN_BATCH", "RNN_CNN_DROP", "RNN_CNN_BATCH_DROP"])
    parser.add_argument("-r", "--rate", default=1e-4, type=float, help="Learning rate. 1e-4 by default")
    parser.add_argument("-f", "--frame", type=int, nargs=2, default=[240, 320], help="Frame size -f Height Width. 240 x 320 by default")
    parser.add_argument("-s", "--step", type=int, default=30, help="Number of sequential frames using by RNN. 30 by default")
    parser.add_argument("-e", "--epoch", type=int, default=5, help="Number of epochs to train. 5 by default")
    parser.add_argument("-b", "--batch", type=int, default=10, help="Batch size. 10 by default")
    parser.add_argument("-d", "--dir", help="Checkpoint directory")
    parser.add_argument("-n", "--norm", default="Normal_0.05", help="Initialization to use for learning from scratch")
    parser.add_argument("--max", type=int, help="Max count of video to use")
    parser.add_argument("--rnn", default=50, type=int, help="Size of RNN state vector. 50 by default")

    args = parser.parse_args()

    source_url = 'https://datastora.blob.core.windows.net/datasets/HockeyFights.zip'
    # source_url = 'http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip'
    source_dir = 'Data'

    frame = (*args.frame, 3)
    dataset = ds.HockeyDataset(source_url, source_dir, frame, max_size=args.max)
    model = create_model(args.type, frame, args.rnn, args.step, norm=args.norm, dir=args.dir)

    if args.mode == "TRAIN":
        trainer = create_trainer(args.type, model, args.rate)
        trainer.train(dataset, epochs=args.epoch, batch_size=args.batch)

    elif args.mode == "PREDICT":
        predictor = create_predictor(model)
        result = predictor.predict(dataset, batch_size=args.batch)
        print("Accuracy:", predictor.accuracy(result))


def create_model(type, frame, state, steps, norm, dir=None):
    def full_model(path):
        return lambda: learning.fullVideo.FullModel(
            frame=frame, norm_type=norm, checkpoint_dir=path,
            seed=100).build(rnn_state=state, avg_result=True)

    def simple_model(path):
        return lambda: learning.simpleVideo.SimpleModel(
            frame=frame, norm_type=norm, checkpoint_dir=path,
            seed=100).build(rnn_state=state, num_steps=steps, avg_result=True)

    def cnn_model(path, batch_norm, dropout):
        return lambda: learning.cnnVideo.CnnModel(
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
        return lambda: learning.cnnVideo.CnnTrainer(model).build(learning_rate=rate)

    trainers = {
        "RNN_FULL": lambda: learning.fullVideo.FullTrainer(model).build(learning_rate=rate),
        "RNN_SIMPLE": lambda: learning.simpleVideo.SimpleTrainer(model).build(learning_rate=rate),
        "RNN_CNN": cnn_trainer(),
        "RNN_CNN_BATCH": cnn_trainer(),
        "RNN_CNN_DROP": cnn_trainer(),
        "RNN_CNN_BATCH_DROP": cnn_trainer(),
    }

    return trainers[type]()


def create_predictor(type, model):
    return learning.Predictor(model)


if __name__ == "__main__":
    main()
