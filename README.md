# Handwritten Digit Recognition

A PyTorch multilayer perceptron trained to recognize MNIST digits. `draw.py` provides a Tkinter canvas for drawing a digit and viewing the model's prediction.

## Run

Install Python with Tkinter support and PyTorch. From this folder:

```sh
python -m pip install torch
python train.py
python test.py
python draw.py
```

`train.py` downloads MNIST data when needed and saves the best model as `mnist_mlp_best.pt`. `test.py` reports test accuracy. Run `draw.py` after training to use the drawing interface.

## Evaluation

Using `train.py` with seed 42, the checkpoint selected by validation accuracy achieved 98.25% accuracy (9,825 of 10,000 images) on the MNIST test set when evaluated by `test.py` in a CPU run. The downloaded dataset and trained checkpoint are generated locally and excluded from Git.
