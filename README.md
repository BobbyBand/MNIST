# 🧠 MNIST Handwritten Digit Recognition with TensorFlow  

## 📌 Overview  
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow** to classify handwritten digits from the **MNIST dataset**. The dataset is loaded from **Kaggle** in CSV format, preprocessed, and trained on a CNN model.  

## 📂 Dataset  
We use the Kaggle MNIST dataset:  
🔗 [MNIST in CSV format](https://www.kaggle.com/datasets/avnishnish/mnist-in-csv)  

- `mnist_train.csv`: 60,000 training images (each row = 784 pixel values + 1 label)  
- `mnist_test.csv`: 10,000 test images  

## 🚀 Installation  
### 1️⃣ **Set up the environment**  
Ensure you have Python installed, then install the required libraries:  
```bash
pip install tensorflow pandas numpy matplotliby
