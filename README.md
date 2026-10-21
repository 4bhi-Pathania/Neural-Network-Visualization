# Neural Network X-Ray Lab

Interactive visualization lab for exploring how a NumPy neural network learns two-dimensional classification problems.

## Highlights

- NumPy implementation of forward propagation, backpropagation, binary cross-entropy, and mini-batch gradient descent.
- XOR, Two Moons, Concentric Circles, Gaussian Blobs, and Spiral datasets.
- Configurable architecture, activation, learning rate, epochs, batch size, and L2 regularization.
- Decision boundaries, learning curves, hidden-neuron activation maps, weights, and experiment comparison.
- Stratified train/validation/test splits, held-out test metrics, confusion matrix, and finite-difference gradient checking.

## Run

```powershell
cd "C:\Users\Abhinandan Pathania\OneDrive\Desktop\neuralnetworks"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name nn-xray-lab --display-name "Python (Neural Network X-Ray)"
jupyter notebook
```

Open `Neural_Network_X_Ray_Final.ipynb`, select **Python (Neural Network X-Ray)**, and choose **Run All**.

## Project structure

```text
Neural_Network_X_Ray_Final.ipynb
nn_xray_lab/       # model, data, evaluation, dashboard
tests/             # automated checks
```

Run tests with:

```powershell
python -m unittest discover -s tests -v
```
