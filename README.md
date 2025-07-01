---

# 📧 Spam Email Classifier using Logistic Regression

This project builds a simple **spam classifier** using a **Logistic Regression** model trained on email content. It uses **TF-IDF** vectorization for feature extraction and demonstrates how classical machine learning can be applied to **text classification** tasks like spam detection.

---

## 🧠 Features

* Preprocesses email text using `TfidfVectorizer`
* Converts spam/ham labels to binary (1 = spam, 0 = ham)
* Trains a logistic regression model on TF-IDF features
* Tests accuracy on unseen data
* Allows manual input for real-time spam prediction

---

## 🛠️ Technologies Used

* Python
* pandas, NumPy
* scikit-learn (TF-IDF, train/test split, LogisticRegression)

---

## 📂 Dataset

This project uses a CSV file named `mail_data.csv` with the following columns:

| Category | Message                           |
| -------- | --------------------------------- |
| spam     | "You've won \$1000..."            |
| ham      | "Hi there, are we meeting today?" |

You can get a sample dataset like this from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) or prepare your own.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone [https://github.com/krisjscott/spam-mail-detection]
cd spam-email-classifier
```

### 2. Install Required Packages

```bash
pip install pandas numpy scikit-learn
```

### 3. Add the Dataset

Place your `mail_data.csv` file in the same directory as the script.

### 4. Run the Script

```bash
python spam_classifier.py
```

---

## 📊 Sample Output

```
![Screenshot 2025-07-01 142512](https://github.com/user-attachments/assets/87729c75-c90c-465a-b62e-650ae42c78bf)


---

## 📬 Try Custom Emails

The script includes an example:

```python
input_your_mail = [
    "this is a test mail, please ignore it",
    "Congratulations! You've won a lottery! Click here to claim your prize."
]
```

It then predicts whether each message is spam or not using the trained model.

---

## 📈 Future Improvements

* Upgrade to deep learning (LSTM, BERT, etc.)
* Add Flask API or Streamlit UI for web interface
* Save model and vectorizer for deployment

---

