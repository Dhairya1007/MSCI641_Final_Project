import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from models.model_logrobust import logrobust_model

# Function to read and parse log files
def read_logs(log_dir):
    logs = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    logs.extend(f.readlines())
    return logs

# Function to preprocess logs
def preprocess_logs(logs):
    processed_logs = []
    for log in logs:
        # Remove timestamps and other non-textual information
        log = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', '', log)
        log = re.sub(r'\[.*?\]', '', log)
        log = re.sub(r'org.apache.hadoop.*? ', '', log)
        log = re.sub(r'\s+', ' ', log).strip()
        processed_logs.append(log)
    return processed_logs

# Function to read anomaly labels
def read_labels(label_file):
    labels_df = pd.read_csv(label_file)
    return labels_df

# Main script
log_dir = './Hadoop'  # Update with actual path
label_file = './Hadoop/anamoly_label.csv'  # Update with actual path

# Read and preprocess logs
logs = read_logs(log_dir)
processed_logs = preprocess_logs(logs)

# Read labels
labels_df = read_labels(label_file)

# Create a DataFrame
data_df = pd.DataFrame({
    'log': processed_logs,
    'label': labels_df['label']
})

# Feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
bow_vectorizer = CountVectorizer(max_features=1000)

tfidf_features = tfidf_vectorizer.fit_transform(data_df['log'])
bow_features = bow_vectorizer.fit_transform(data_df['log'])

# Train-test split
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_features, data_df['label'], test_size=0.2, random_state=42)
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_features, data_df['label'], test_size=0.2, random_state=42)

# Logistic Regression model
tfidf_model = LogisticRegression(max_iter=1000)
tfidf_model.fit(X_train_tfidf, y_train_tfidf)

bow_model = LogisticRegression(max_iter=1000)
bow_model.fit(X_train_bow, y_train_bow)

# LogRobust model
logrobust_model = logrobust_model()
logrobust_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train-test split for LogRobust model
X_train_logrobust, X_test_logrobust, y_train_logrobust, y_test_logrobust = train_test_split(tfidf_features.toarray(), data_df['label'], test_size=0.2, random_state=42)

logrobust_model.fit(X_train_logrobust, y_train_logrobust, epochs=10, batch_size=32, validation_split=0.2)

# Predictions and evaluation
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)
y_pred_bow = bow_model.predict(X_test_bow)

print("TF-IDF Vectorizer:")
print("Accuracy Score:", accuracy_score(y_test_tfidf, y_pred_tfidf))
print("Confusion Matrix:\n", confusion_matrix(y_test_tfidf, y_pred_tfidf))
print("Classification Report:\n", classification_report(y_test_tfidf, y_pred_tfidf))

print("Bag of Words Vectorizer:")
print("Accuracy Score:", accuracy_score(y_test_bow, y_pred_bow))
print("Confusion Matrix:\n", confusion_matrix(y_test_bow, y_pred_bow))
print("Classification Report:\n", classification_report(y_test_bow, y_pred_bow))

# Evaluate LogRobust model
y_pred_logrobust = logrobust_model.predict(X_test_logrobust)
y_pred_logrobust = (y_pred_logrobust > 0.5).astype(int)

print("LogRobust Model:")
print("Accuracy Score:", accuracy_score(y_test_logrobust, y_pred_logrobust))
print("Confusion Matrix:\n", confusion_matrix(y_test_logrobust, y_pred_logrobust))

