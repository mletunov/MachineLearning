import learning
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Network functional mode", required=True, choices=["TRAIN", "PREDICT", "WEB"])
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
    
    frame = (*args.frame, 3)
    model = learning.factory.create_model(args.type, frame, args.rnn, args.step, norm=args.norm, dir=args.dir)

    if args.mode == "TRAIN":
        trainer = learning.factory.create_trainer(args.type, model, args.rate)
        dataset = learning.factory.local_dataset(frame, args.max)
        trainer.train(dataset, epochs=args.epoch, batch_size=args.batch)
    
    elif args.mode == "PREDICT":
        predictor = learning.factory.create_predictor(args.type, model)
        dataset = learning.factory.local_dataset(frame, args.max)
        result = predictor.predict(dataset, batch_size=args.batch)
        print("Accuracy:", predictor.accuracy(result))
    
    elif args.mode == "WEB":
        from os import environ
        from api import app

        HOST = environ.get('SERVER_HOST', 'localhost')
        app.config['predictor'] = learning.factory.create_predictor(args.type, model)
        try:
            PORT = int(environ.get('SERVER_PORT', '5555'))
        except ValueError:
            PORT = 5555
        app.run(HOST, PORT)

if __name__ == "__main__":
    main()
