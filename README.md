# CNN-Py-MNIST

A convolutional neural network built entirely from scratch in NumPy — no PyTorch, no TensorFlow, no ML framework of any kind. Implements forward and backward propagation manually for every layer, trained on the MNIST handwritten digit dataset.

---

## Architecture

```
Input (1×28×28)
  → Conv2D (8 filters, 3×3, padding=1)  → ReLU  → MaxPool (2×2)
  → Conv2D (16 filters, 3×3, padding=1) → ReLU  → MaxPool (2×2)
  → Flatten
  → Dense (392 → 64)                    → ReLU
  → Dense (64 → 10)
  → Softmax + Cross-Entropy Loss
```

Each layer is implemented as a standalone module with explicit forward and backward passes. Gradients are computed analytically — no autograd.

---

## Project Structure

```
CNN-Py-MNIST/
├── ConvolutionLayer.py   # Conv2D forward & backward (stride, padding support)
├── maxpool.py            # MaxPool forward & backward
├── flatten_layer.py      # Flatten forward & backward
├── denselayer.py         # Fully-connected layer forward & backward
├── Relu.py               # ReLU activation forward & backward
├── Softmax_Loss.py       # Softmax + categorical cross-entropy (fused for numerical stability)
├── load_dataset.py       # MNIST loader from raw IDX binary format (no dataset library)
├── training.py           # Training loop with mini-batch SGD
├── predict.py            # Inference module (imported by app.py)
├── app.py                # Streamlit UI — drawable canvas + file upload
└── requirements.txt      # numpy, streamlit, streamlit-drawable-canvas, pillow
```

---

## Running the App

```bash
git clone https://github.com/dhruv-gupta-0311/CNN-Py-MNIST
cd CNN-Py-MNIST
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

The app lets you draw a digit on a canvas or upload an image. It shows the prediction, confidence score, and a probability breakdown across all 10 classes.

---

## Training From Scratch

```bash
python training.py
```

Hyperparameters (configurable at the top of `training.py`):

| Parameter     | Default |
|---------------|---------|
| Learning rate | 1e-3    |
| Epochs        | 20      |
| Batch size    | 32      |
| Training samples | 10,000 |

MNIST is downloaded automatically on first run into `mnist_data/` directly from the IDX binary source — no dataset library required.

Trained weights are saved to `mnist_model.npz` after training completes.

---

## Known Limitations

- **Trained on 10,000 of 60,000 available samples** due to local compute constraints. Accuracy is functional but below what full training achieves. To train on the full dataset, remove the slicing in `training.py`:
  ```python
  # Change this:
  X_train_small = X_train[:10000]
  # To this:
  X_train_small = X_train
  ```
- **No optimizer beyond vanilla SGD.** Momentum or Adam would improve convergence speed and final accuracy.
- **Canvas-drawn digits** may have lower accuracy than clean scanned images due to stroke thickness and centering differences from the training distribution.

---

## Implementation Notes

Backpropagation is implemented manually for every layer. The conv layer computes gradients with respect to filters, biases, and inputs using explicit loop-based correlation. MaxPool backward routes gradients only through the positions that held the maximum value in the forward pass. The Softmax and cross-entropy loss are fused into a single layer to avoid numerical instability from computing them separately.

MNIST data is loaded by reading the raw IDX binary format directly using `numpy.frombuffer` — the image file header is 16 bytes, label file header is 8 bytes, everything after is raw data.

---

## Future Work

- Train on full 60k sample set (compute bottleneck only)
- Add momentum or Adam optimiser
- Extend to civic issue detection (potholes, garbage, broken streetlights) using a real-world image dataset