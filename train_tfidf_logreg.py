"""
İlk sentiment modeli: TF-IDF + Logistic Regression.

Eğitim kümesi: dataset/train_sentiment.csv
Test kümesi:    dataset/test_sentiment.csv

Girdi yalnızca `comment`, hedef yalnızca `sentiment`.
TF-IDF yalnızca eğitim yorumlarına fit edilir; test kümesine sadece transform uygulanır.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Dosya yolları. Ham ve dengeli CSV'lere dokunulmaz.
TRAIN_PATH = Path("dataset/train_sentiment.csv")
TEST_PATH = Path("dataset/test_sentiment.csv")
MODEL_DIR = Path("models")
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "logreg_sentiment.joblib"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix_tfidf_logreg.png"

# Raporlarda sınıflar her zaman aynı sırada görünsün.
CLASS_ORDER = ["Negative", "Neutral", "Positive"]


def load_split(path: Path) -> pd.DataFrame:
    """CSV'yi okur ve modelin kullanacağı iki sütunun dolu olduğunu kontrol eder."""
    df = pd.read_csv(path)
    required = {"comment", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} içinde eksik sütunlar: {sorted(missing)}")
    if df["comment"].isna().any() or df["sentiment"].isna().any():
        raise ValueError(f"{path} içinde boş comment veya sentiment var.")
    return df


def main() -> None:
    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)

    # point, category, seller ve diğer sütunlar bilinçli olarak alınmaz.
    # Duygu etiketi zaten yıldızdan üretildiği için point modele verilirse
    # aynı bilgi ikinci kez sayılmış olur.
    X_train = train_df["comment"].astype(str)
    y_train = train_df["sentiment"].astype(str)
    X_test = test_df["comment"].astype(str)
    y_test = test_df["sentiment"].astype(str)

    print("=== Veri ===")
    print(f"Eğitim: {len(X_train)} satır | Test: {len(X_test)} satır")
    print("Eğitim sınıf dağılımı:")
    print(y_train.value_counts().reindex(CLASS_ORDER).to_string())
    print("Test sınıf dağılımı:")
    print(y_test.value_counts().reindex(CLASS_ORDER).to_string())

    # İlk deneyde ek metin temizliği yok: emoji, noktalama ve yazım olduğu gibi kalır.
    # Varsayılan token deseni Unicode sözcük karakterlerini kullanır; ç, ğ, ı, ö, ş, ü
    # bu desende harf sayılır ve atılmaz. lowercase=True, sklearn'in varsayılanıdır.
    # min_df=2 tek seferlik yazım hatalarını seyreltir. (1, 2) n-gram, tek kelimeyle
    # birlikte "çok güzel" gibi kısa ikilileri de tutar. Üst sınır bellek ve süreyi
    # öngörülebilir kılar.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        token_pattern=r"(?u)\b\w\w+\b",
    )

    # Fit yalnızca eğitim metinlerinde. Test metinleri sözlüğe ve IDF ağırlıklarına girmez.
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\n=== TF-IDF ===")
    print(f"Sözlük boyutu: {len(vectorizer.vocabulary_)}")
    print(f"Eğitim matrisi: {X_train_tfidf.shape} | Test matrisi: {X_test_tfidf.shape}")

    # Üç sınıf dengeli olduğu için class_weight kullanılmadı.
    # C=1.0 ve lbfgs, sklearn'in çok sınıflı lojistik regresyon varsayılanlarıdır.
    # max_iter, seyrek TF-IDF üzerinde varsayılan 100 adımın yetmemesi ihtimaline karşı yükseltildi.
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(X_train_tfidf, y_train)

    y_pred = classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", labels=CLASS_ORDER)
    recall_macro = recall_score(y_test, y_pred, average="macro", labels=CLASS_ORDER)
    f1_macro = f1_score(y_test, y_pred, average="macro", labels=CLASS_ORDER)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", labels=CLASS_ORDER)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", labels=CLASS_ORDER)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=CLASS_ORDER)

    print("\n=== Genel metrikler (test) ===")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision (macro):  {precision_macro:.4f}")
    print(f"Recall (macro):     {recall_macro:.4f}")
    print(f"F1 (macro):         {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted):    {recall_weighted:.4f}")
    print(f"F1 (weighted):        {f1_weighted:.4f}")

    print("\n=== Sınıf bazlı rapor ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASS_ORDER,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=[f"Gerçek {c}" for c in CLASS_ORDER], columns=[f"Tahmin {c}" for c in CLASS_ORDER])
    print("=== Confusion matrix ===")
    print(cm_df.to_string())

    # Grafik, etkileşimli pencere açmadan dosyaya yazılır.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("TF-IDF + Logistic Regression")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = int(cm[row, col])
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix grafiği: {CONFUSION_MATRIX_PATH}")

    # Vektörleştirici ve sınıflandırıcı ayrı kaydedilir; birlikte yüklenmeden tahmin üretilemez.
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    print(f"TF-IDF kaydedildi: {VECTORIZER_PATH}")
    print(f"Model kaydedildi:  {CLASSIFIER_PATH}")

    # Aynı 10 satırın tekrar seçilebilmesi için tohum sabit.
    sample = test_df.loc[:, ["comment", "sentiment"]].copy()
    sample["prediction"] = y_pred
    sample = sample.sample(n=10, random_state=42)
    print("\n=== Test kümesinden 10 örnek ===")
    for index, row in sample.iterrows():
        correct = row["sentiment"] == row["prediction"]
        verdict = "doğru" if correct else "yanlış"
        comment = str(row["comment"]).replace("\n", " ")
        if len(comment) > 180:
            comment = comment[:177] + "..."
        print(f"- satır {index} | gerçek: {row['sentiment']} | tahmin: {row['prediction']} | {verdict}")
        print(f"  yorum: {comment}")


if __name__ == "__main__":
    main()
