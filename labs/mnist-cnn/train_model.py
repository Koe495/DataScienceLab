"""Train a CNN and evaluate the selected checkpoint on the MNIST test set."""

import argparse

from model import TRAINED_MODEL, load_splits, train_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    train, validation, test = load_splits()
    model, _ = train_model(train, validation, epochs=args.epochs)
    metrics = model.evaluate(*test, verbose=0, return_dict=True)
    print(f"Test accuracy: {metrics['accuracy']:.2%}")
    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Checkpoint: {TRAINED_MODEL}")


if __name__ == "__main__":
    main()
