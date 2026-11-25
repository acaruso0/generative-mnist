# Toy model: generative architectures on MNIST
This is just a toy project for which I implemented two architectures, diffusion and flow matching, trained and tested on the MNIST dataset.
The generative architectures are validated at each epoch with a small pre-trained classifier.

Remember to first sync your environment with 'uv sync'!

## Training the classifier
uv run python -m generative_mnist.train_classifier

## Training the generative model
uv run python -m generative_mnist.train --flow --classifier "ckpts/classifier/20261125_174016_10"

## Predict and visualize a 5x10 grid of conditioned labels
uv run python -m generative_mnist.predict "ckpts/diffusion_20261124_171948/diffusion_20261124_171948_100"
