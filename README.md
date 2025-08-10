# Assignment 2 – Support Vector Machines, Neural Networks, PCA, and Regression

## 📄 Summary

This assignment explores a range of machine learning techniques including **Support Vector Machines (SVM)**, **Logistic Regression**, **Neural Networks (MLP & CNN)**, **Principal Component Analysis (PCA)**, and **Regression methods** (OLS, Ridge, and SVR).  
The work is divided into three main parts:

---

### **Part 1 – Support Vector Machines (SVM)**

- Used dataset provided via oracle functions for binary/multiclass classification tasks.  
- Implemented both **linear** and **kernelized** SVMs (RBF kernel).  
- Applied grid search to tune hyperparameters \( C \) and \( \gamma \).  
- Evaluated using accuracy, precision, recall, F1-score, and confusion matrices.  
- Visualized decision boundaries for low-dimensional projections.  

---

### **Part 2 – Logistic Regression, MLP, CNN & PCA (MNIST-JPG)**

- **Dataset**: MNIST-JPG subset (10,000 images; 28×28 grayscale; 10 classes) retrieved using SR number-specific oracle.  
- Implementations:
  - **MLP**: Fully connected network with flattened 784-dim input; trained to predict class probabilities.  
  - **CNN**: Convolutional network trained directly on image data for digit classification.  
  - **PCA**: Extracted dimensionality-reduced features; reconstructed images using varying numbers of principal components.  
  - **MLP with PCA**: MLP trained on PCA-extracted features.  
  - **Logistic Regression with PCA**: Multi-class classifier + 10 binary one-vs-rest classifiers; computed ROC curves and AUC scores.  
- Performance evaluation:
  - Confusion matrices for all models.  
  - Accuracy, precision, recall, and F1-score per class.  
  - Average AUC from ROC curves of binary classifiers.  

---

### **Part 3 – Regression (OLS, Ridge, SVR)**

- **Linear Regression**:
  - Queried two datasets from oracle; solved **Ordinary Least Squares** and **Ridge Regression** (\( \lambda = 1 \)).
  - Calculated optimal weights \( w_{\text{ols}} \) and \( w_{\text{rr}} \), reported MSE on training data.  
  - Saved learned weights as CSV files in the required format.  
- **Support Vector Regression (SVR)**:
  - Stock prediction task using assigned stock from StockNet dataset.  
  - Preprocessed data with normalization and sliding time-window features.  
  - Implemented:
    - **Slack Linear SVR** (dual form via CVXOPT).  
    - **Kernelized SVR** with RBF kernel for \( \gamma \in \{1, 0.1, 0.01, 0.001\} \).  
  - Trained models for time windows \( t \in \{7, 30, 90\} \).  
  - Plotted predicted prices, actual prices, and moving averages on test data.  

---

### **📊 Key Results & Observations**

- **SVM**: RBF kernel achieved higher accuracy than linear SVM, especially in non-linearly separable cases.  
- **Neural Networks**:
  - CNN outperformed MLP on MNIST-JPG due to its ability to capture spatial features.  
  - PCA-based MLP and logistic regression showed reduced accuracy but improved computational efficiency.  
- **PCA**:
  - Visual inspection showed that ~50 components preserved most of the image structure, balancing reconstruction quality and dimensionality.  
- **Regression**:
  - Ridge regression slightly reduced overfitting compared to OLS.  
  - Kernel SVR captured non-linear trends in stock data more effectively than linear SVR, particularly for longer time windows.

---

### **🛠 Tools & Libraries Used**

- **Python 3.12.8**  
- `numpy`, `scipy`, `pandas`  
- `scikit-learn`, `cvxopt`, `torch`, `torchvision`  
- `matplotlib`, `seaborn`
