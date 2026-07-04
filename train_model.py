import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# =========================
# VECTORIZATION
# =========================
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# MODEL TRAINING
# =========================
model = SVC(probability=True)
model.fit(X_train_vec, y_train)

# =========================
# PREDICTIONS
# =========================
y_pred = model.predict(X_test_vec)

# =========================
# EVALUATION
# =========================
print("\n🔹 MODEL PERFORMANCE 🔹\n")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:\n")
report_text = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
print(report_text)

print("📉 Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =========================
# 📊 CONFUSION MATRIX PLOT
# =========================
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# =========================
# 📊 CLASSIFICATION REPORT PLOT
# =========================
report_dict = classification_report(
    y_test, y_pred,
    target_names=["Ham", "Spam"],
    output_dict=True
)

df_report = pd.DataFrame(report_dict).transpose()

df_report.iloc[:-1, :-1].plot(kind="bar")
plt.title("Classification Report (Precision / Recall / F1)")
plt.xlabel("Classes")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend()
plt.show()

# =========================
# 🔁 PROFESSIONAL DFD (CLEAN UI)
# =========================
G = nx.DiGraph()

nodes = [
    "User Input",
    "Flask Web App",
    "Text Preprocessing",
    "TF-IDF Vectorizer",
    "SVM Classifier",
    "Prediction Output",
    "Result Dashboard"
]

edges = [
    ("User Input", "Flask Web App"),
    ("Flask Web App", "Text Preprocessing"),
    ("Text Preprocessing", "TF-IDF Vectorizer"),
    ("TF-IDF Vectorizer", "SVM Classifier"),
    ("SVM Classifier", "Prediction Output"),
    ("Prediction Output", "Result Dashboard")
]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Horizontal layout
pos = {
    "User Input": (0, 0),
    "Flask Web App": (2, 0),
    "Text Preprocessing": (4, 0),
    "TF-IDF Vectorizer": (6, 0),
    "SVM Classifier": (8, 0),
    "Prediction Output": (10, 0),
    "Result Dashboard": (12, 0)
}

plt.figure(figsize=(18, 3))

# Draw nodes as rectangles (using node_shape='s')
nx.draw_networkx_nodes(
    G, pos,
    node_shape='s',
    node_color="#60a5fa",   # single light color
    node_size=4000,
    edgecolors="black",
    linewidths=1.5
)

# Draw edges (arrows)
nx.draw_networkx_edges(
    G, pos,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=20,
    width=2,
    edge_color="black"
)

# Labels
nx.draw_networkx_labels(
    G, pos,
    font_size=10,
    font_weight="bold",
    font_color="black"
)

# Background clean white
plt.gca().set_facecolor("white")

plt.title("SpamGuard AI - Data Flow Diagram", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
# =========================
# SAVE MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Model trained and saved successfully!")