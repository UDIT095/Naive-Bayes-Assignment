# 📝 Text Classification and Sentiment Analysis using Naive Bayes

## 📌 Project Title
**Text Classification using Naive Bayes and Sentiment Analysis on Blog Posts**

## 📖 Overview
This project focuses on:
- Categorizing blog posts using a **Naive Bayes text classification model**.
- Performing **sentiment analysis** to evaluate the emotional tone (positive, negative, neutral) of each post.

It demonstrates foundational techniques in **Natural Language Processing (NLP)** including preprocessing, vectorization (TF-IDF), and sentiment analysis using `TextBlob`.

---

## 📂 Dataset
- **File:** `blogs.csv`  
- **Columns:**
  - `Data`: Text content of blog posts.
  - `Labels`: Category of each post (e.g., religion, politics, autos).

---

## 📁 Repository Contents
- `blogs.csv` – Dataset containing blog posts and categories  
- `Naive Bayes and Text Mining.ipynb` – Jupyter notebook with full implementation  
- `NLP and Naive Bayes.docx` – Assignment and project overview  

---

## 🛠️ Tools and Libraries
- **Python**, **Jupyter Notebook**
- `pandas`, `scikit-learn`, `nltk`, `textblob`, `string`

---

## 🔄 Project Workflow

### 1. 🧹 Data Preprocessing
- Load and explore the dataset
- Clean and tokenize the text
- Remove stopwords and punctuation
- Convert text to numerical format using **TF-IDF**

### 2. 🤖 Naive Bayes Classification
- Split data into training and test sets
- Train a **MultinomialNB** model
- Evaluate performance using:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1-score)

### 3. 💬 Sentiment Analysis
- Use `TextBlob` to classify each blog post as:
  - Positive
  - Negative
  - Neutral
- Analyze sentiment distribution across different categories

---

## ✅ Key Results

- **Model Accuracy:** ~73%
- **Best Performing Categories:** `comp.os.ms-windows.misc`, `sci.space`
- **Challenging Categories:** `talk.religion.misc`, `comp.windows.x`
- **Sentiment Trends:**
  - Most blog posts reflect **positive sentiment**
  - Sentiment varies by topic (e.g., religious/political topics tend to skew negative)

---

## ▶️ How to Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Required Libraries**  
   ```bash
   pip install pandas scikit-learn nltk textblob
   ```

3. **Download NLTK Resources**  
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Run the Jupyter Notebook**  
   ```bash
   jupyter notebook
   ```

   Open `Naive Bayes and Text Mining.ipynb` and run the cells sequentially.

---

## 🧠 Learning Highlights
- End-to-end implementation of text classification using **Naive Bayes**
- Applied **TF-IDF** vectorization and **NLTK**-based preprocessing
- Performed real-world **sentiment analysis**
- Understood common NLP challenges like class imbalance and ambiguity
