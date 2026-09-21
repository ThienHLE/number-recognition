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

The downloaded dataset and trained checkpoint are generated files and are excluded from Git. The copy already on this computer remains available locally.
