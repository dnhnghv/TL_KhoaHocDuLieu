# train_sentiment.py

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data():
   
    data_path = r"C:\DINHNGUYENHOANGVU\TieuLuan_KHDL\data\sentiment_data.csv"
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    # mapping nhãn
    mapping = {'tiêu cực': 0, 'trung tính': 1, 'tích cực': 2}
    df['label_num'] = df['label'].map(mapping)
    return df

def vectorize(texts, method='tfidf'):
    if method == 'count':
        vec = CountVectorizer(ngram_range=(1,2), min_df=2)
    else:
        vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=2)
    X = vec.fit_transform(texts)
    return vec, X

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {}
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results['nb'] = {
        'model': nb,
        'report': classification_report(y_test, y_pred_nb, 
                    target_names=['tiêu cực','trung tính','tích cực'], zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred_nb)
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['lr'] = {
        'model': lr,
        'report': classification_report(y_test, y_pred_lr, 
                    target_names=['tiêu cực','trung tính','tích cực'], zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred_lr)
    }
    return results

def main():
    # 1. Load dữ liệu
    df = load_data()
    texts = df['comments'].astype(str)
    labels = df['label_num']

    # 2. Vectorize
    vec, X = vectorize(texts, method='tfidf')

    # 3. Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 4. Train & evaluate
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    for name, info in results.items():
        print(f"\n===== {name.upper()} =====")
        print(f"Accuracy: {info['accuracy']:.4f}")
        print(info['report'])

    # 5. Lưu mô hình tốt nhất (Logistic Regression)
    best_model = results['lr']['model']
    pipeline = {'vectorizer': vec, 'model': best_model}
    model_path = r"C:\DINHNGUYENHOANGVU\TieuLuan_KHDL\sentiment_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Lưu mô hình vào {model_path}")

if __name__ == '__main__':
    main()
