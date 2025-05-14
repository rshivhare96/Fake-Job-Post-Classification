# 🎯 Fake Job Post Classification

This project focuses on identifying **fraudulent job postings** using machine learning and NLP techniques. By analyzing thousands of job listings scraped from the internet, the goal is to create a classification model that flags potentially deceptive or scam job ads — helping job seekers avoid fraud.

---

## 📦 Dataset Overview

The dataset contains **17,880 job postings**, with a binary classification target:

- `0` – Legitimate job post  
- `1` – Fraudulent job post  

### 📌 Class Distribution

| Class | Description       | Count  | Percentage |
|-------|-------------------|--------|------------|
| 0     | Real Jobs         | 17,014 | 95.16%     |
| 1     | Fake Jobs         |   866  |  4.84%     |

> ⚠️ The dataset is highly **imbalanced**, so **accuracy alone is not a reliable metric**.

---

## 🔎 Data Cleaning & Preprocessing

### 🧹 Steps Taken:
- Dropped irrelevant columns (e.g., `job_id`)  
- Handled missing values in text fields using empty strings  
- Created features:
  - Word count and character count for `company_profile`, `description`, `requirements`, `benefits`  
- Converted `fraudulent` column to integer for modeling  
- Cleaned text by:
  - Lowercasing  
  - Removing punctuation, stopwords, and digits  
  - Tokenization and whitespace trimming  

---

## 📊 Exploratory Data Analysis

### 📈 Character & Word Counts

| Field            | Real Jobs (Mean Words) | Fake Jobs (Mean Words) | Real Jobs (Mean Chars) | Fake Jobs (Mean Chars) |
|------------------|------------------------:|-----------------------:|-----------------------:|-----------------------:|
| Company Profile  |                     238 |                    133 |                  1442  |                   739  |
| Description      |                    1015 |                   1002 |                  6280  |                  6179  |
| Requirements     |                     569 |                    553 |                  3546  |                  3438  |
| Benefits         |                     287 |                    276 |                  1791  |                  1735  |

> 🔍 **Fake job posts** tend to have **significantly shorter company profiles**. Descriptions and requirements are similar in length across both classes.

### 📊 N-Gram Analysis
- **Most frequent unigrams** in both real and fake posts include general-purpose tokens (`work`, `experience`, `job`, etc.)  
- Common use of punctuation and generic language in both categories  
- Suggests that raw unigrams are **not discriminative enough** without deeper semantic modeling  

---

## 🔠 Text Vectorization & Modeling

### TF-IDF Vectorization
- Used `TfidfVectorizer` on the `description` field  
- Converted text into sparse feature vectors (unigrams only)  

---

## 🤖 Machine Learning Models

### 🔹 1. Logistic Regression (TF-IDF)

| Metric     | Score    |
|------------|---------:|
| Accuracy   |   96.3%  |
| Precision  |    0.86  |
| Recall     |    0.70  |
| F1 Score   |    0.77  |
| ROC-AUC    | **0.85** |

> ✅ Strong performance with good interpretability  
> ⚠️ Some false negatives due to class imbalance  

---

### 🔹 2. Deep Learning (GloVe Embeddings + Feedforward Neural Network)

- Used **200-dimensional GloVe embeddings**  
- Input: padded sequences of tokenized `description`  
- Model: 2 Dense layers + ReLU + Dropout  

| Metric     | Score     |
|------------|----------:|
| Accuracy   | **97.9%** |
| ROC-AUC    |    0.81   |

> High accuracy but **lower AUC** suggests overfitting to the majority class  

---

### 🔹 3. BERT Embeddings + Logistic Regression

- Used pretrained **BERT sentence embeddings**  
- Applied to a **2,000-sample subset** due to processing time  

| Metric  | Score  |
|---------|-------:|
| ROC-AUC | 0.77   |

> Promising approach but requires full dataset for improved results  

---

## 🧠 Model Comparison

| Model                  | Accuracy | ROC-AUC | Precision | Recall | F1 Score |
|------------------------|---------:|--------:|----------:|-------:|---------:|
| Logistic Regression    |   96.3%  |   0.85  |     0.86  |  0.70  |    0.77  |
| Deep Learning (GloVe)  |   97.9%  |   0.81  |       –   |    –   |      –   |
| BERT + Logistic Regr.  |      –   |   0.77  |       –   |    –   |      –   |

> 🏆 **Logistic Regression** with TF-IDF emerged as the **most balanced and effective model** in terms of AUC and interpretability.

---

## 📌 Key Insights

- ⚠️ **Fake jobs** often have **shorter and less informative company profiles**  
- 📉 GloVe and BERT embeddings underperformed compared to TF-IDF + Logistic Regression due to:
  - Dataset size constraints  
  - Imbalance challenges  
- ✅ **TF-IDF + Logistic Regression** provided a high-AUC, fast, and explainable model  
- 📊 Visualization of predictions showed clear separation between classes in feature space  

---

## 🚀 Future Improvements

- Fine-tune **transformer-based models (BERT, RoBERTa)** on the full dataset  
- Use **class weighting**, **SMOTE**, or **focal loss** to handle imbalance  
- Extract and use **domain-specific keywords** or suspicious patterns (e.g., “pay to apply”, “no experience needed”)  
- Build a **Streamlit or Flask app** for real-time job screening  
- Evaluate on **more recent job posting data** to capture new fraud tactics  

---

## ✅ Conclusion

This project demonstrates that **text-based features**, when carefully preprocessed and vectorized, can effectively identify fraudulent job listings. With continued development, such tools can aid job seekers and platforms in maintaining safe and trustworthy job marketplaces.
