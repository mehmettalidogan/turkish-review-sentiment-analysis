"""
Kontrollü preprocessing deneyi: TF-IDF + Logistic Regression.

Baseline betiğine, kayıtlı baseline modele ve vektörleştiriciye yazılmaz.
Aynı eğitim/test dosyaları ve aynı model parametreleri kullanılır.
TF-IDF yalnızca eğitim metnine fit edilir.
"""

from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

TRAIN_PATH = Path("dataset/train_sentiment.csv")
TEST_PATH = Path("dataset/test_sentiment.csv")
MODEL_DIR = Path("models")

# Baseline kayıtları sadece metrik karşılaştırması için okunur.
BASELINE_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
BASELINE_CLASSIFIER_PATH = MODEL_DIR / "logreg_sentiment.joblib"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer_preprocessed.joblib"
CLASSIFIER_PATH = MODEL_DIR / "logreg_sentiment_preprocessed.joblib"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix_tfidf_logreg_preprocessed.png"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"

CLASS_ORDER = ["Negative", "Neutral", "Positive"]

# Yorum içindeki bağlantılar. Bu kümede comment sütununda URL yok; adım yine de uygulanır.
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Aynı noktalama işaretinin tekrarı tek işaret olur: "!!!" -> "!". Harfler ve emoji durur.
REPEAT_PUNCT_RE = re.compile(r"([!?.,;:…])\1+")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def turkish_lower(text: str) -> str:
    """
    Türkçe küçük harf. str.lower() 'İ' harfini 'i' + birleşik nokta yapar, 'I' harfini 'i' yapar.
    Önce İ→i ve I→ı çevrilir; ç, ğ, ö, ş, ü ve diğer harfler silinmez.
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


def preprocess_comment(text: str) -> str:
    """
    Kontrollü temizlik. Stopword, ek atma ve emoji silme yok.
    'değil', 'yok', 'hiç', 'ama' kelime olduğu için bu fonksiyon onları çıkarmaz.
    """
    cleaned = URL_RE.sub(" ", str(text))
    cleaned = turkish_lower(cleaned)
    cleaned = REPEAT_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def make_vectorizer() -> TfidfVectorizer:
    """train_tfidf_logreg.py içindeki baseline parametrelerinin aynısı."""
    return TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        token_pattern=r"(?u)\b\w\w+\b",
    )


def make_classifier() -> LogisticRegression:
    """train_tfidf_logreg.py içindeki baseline parametrelerinin aynısı."""
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"comment", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} içinde eksik sütunlar: {sorted(missing)}")
    return df


def metric_row(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict:
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=CLASS_ORDER),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=CLASS_ORDER),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=CLASS_ORDER)
    for label, score in zip(CLASS_ORDER, per_class):
        row[f"f1_{label.lower()}"] = float(score)
    return row


def save_confusion_matrix(cm: pd.DataFrame, path: Path) -> None:
    labels = [column.removeprefix("Tahmin ") for column in cm.columns]
    values = cm.to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Preprocessed TF-IDF + Logistic Regression")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = int(values[row, col])
            color = "white" if value > values.max() / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    # Temizlik kelime silmediğini örneklerle gösterir.
    probes = {
        "DEĞİL": "değil",
        "YOK": "yok",
        "HİÇ": "hiç",
        "AMA": "ama",
        "ÇĞIİÖŞÜ": "çğ\u0131iöşü",
        "harika!!!   değil": "harika! değil",
        "bak https://ornek.com/a burada": "bak burada",
        "güzel 😍😍": "güzel 😍😍",
    }
    print("=== Preprocessing kontrolleri ===")
    for raw, expected in probes.items():
        got = preprocess_comment(raw)
        print(f"  {raw!r} -> {got!r}")
        if got != expected:
            raise ValueError(f"Beklenen {expected!r}, gelen {got!r}")

    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)
    y_train = train_df["sentiment"].astype(str)
    y_test = test_df["sentiment"].astype(str)

    x_train_raw = train_df["comment"].astype(str)
    x_test_raw = test_df["comment"].astype(str)
    x_train = x_train_raw.map(preprocess_comment)
    x_test = x_test_raw.map(preprocess_comment)

    changed_train = int((x_train_raw != x_train).sum())
    changed_test = int((x_test_raw != x_test).sum())
    url_train = int(x_train_raw.str.contains(URL_RE).sum())
    url_test = int(x_test_raw.str.contains(URL_RE).sum())
    print("\n=== Metin değişimi ===")
    print(f"Yorumu değişen eğitim satırı: {changed_train} / {len(x_train)}")
    print(f"Yorumu değişen test satırı:    {changed_test} / {len(x_test)}")
    print(f"URL içeren comment: eğitim {url_train}, test {url_test}")

    # Noktalama ve boşluk, kelime belirtecinin zaten attığı karakterler.
    # Karşılaştırma, temizliğin gerçekten farklı kelime dizisi üretip üretmediğini gösterir.
    def token_string(text: str) -> str:
        return " ".join(TOKEN_RE.findall(text.lower()))

    raw_tokens = x_test_raw.map(token_string)
    new_tokens = x_test.map(lambda text: " ".join(TOKEN_RE.findall(text)))
    token_diff = int((raw_tokens != new_tokens).sum())
    print(f"Testte kelime dizisi değişen satır: {token_diff}")

    # Fit yalnızca temizlenmiş eğitim yorumlarında.
    vectorizer = make_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    print("\n=== Yeni TF-IDF ===")
    print(f"Sözlük boyutu: {len(vectorizer.vocabulary_)}")
    print(f"Eğitim matrisi: {x_train_tfidf.shape} | Test matrisi: {x_test_tfidf.shape}")

    classifier = make_classifier()
    classifier.fit(x_train_tfidf, y_train)
    y_pred = classifier.predict(x_test_tfidf)

    # Baseline metrikleri kayıtlı nesnelerden, ham metin üzerinde yeniden hesaplanır.
    baseline_vectorizer = joblib.load(BASELINE_VECTORIZER_PATH)
    baseline_classifier = joblib.load(BASELINE_CLASSIFIER_PATH)
    baseline_pred = baseline_classifier.predict(baseline_vectorizer.transform(x_test_raw))

    comparison = pd.DataFrame(
        [
            metric_row("baseline_tfidf_logreg", y_test, baseline_pred),
            metric_row("preprocessed_tfidf_logreg", y_test, y_pred),
        ]
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    print("\n=== Karşılaştırma ===")
    printable = comparison.copy()
    for column in printable.columns:
        if column != "model":
            printable[column] = printable[column].map(lambda value: f"{value:.4f}")
    print(printable.to_string(index=False))
    print(f"\nKarşılaştırma dosyası: {COMPARISON_PATH}")

    baseline_f1 = float(comparison.loc[comparison["model"] == "baseline_tfidf_logreg", "macro_f1"].iloc[0])
    new_f1 = float(comparison.loc[comparison["model"] == "preprocessed_tfidf_logreg", "macro_f1"].iloc[0])
    delta = new_f1 - baseline_f1
    # 0.005 macro F1, 5400 test satırında yaklaşık 27 örneklik bir kayma demektir.
    # Bunun altındaki fark, Neutral karışmasını kapatmış sayılmaz.
    if delta >= 0.005:
        verdict = "Preprocessing modeli baseline'dan belirgin biçimde iyi; macro F1 en az 0.005 arttı."
    elif delta > 0:
        verdict = (
            "Preprocessing modeli baseline'dan daha iyi değil. Macro F1 sayı olarak "
            f"{delta:.4f} yüksek, fakat fark pratikte yok denecek kadar küçük."
        )
    elif delta < 0:
        verdict = "Preprocessing modeli baseline'dan daha iyi değil; macro F1 daha düşük."
    else:
        verdict = "Preprocessing modeli baseline ile aynı macro F1 değerinde; iyileşme yok."
    print(verdict)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Gerçek {label}" for label in CLASS_ORDER],
        columns=[f"Tahmin {label}" for label in CLASS_ORDER],
    )
    print("\n=== Preprocessing confusion matrix ===")
    print(cm_df.to_string())
    save_confusion_matrix(cm_df, CONFUSION_MATRIX_PATH)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(classifier, CLASSIFIER_PATH)
    print(f"\nTF-IDF kaydedildi: {VECTORIZER_PATH}")
    print(f"Model kaydedildi:  {CLASSIFIER_PATH}")
    print(f"Confusion matrix:  {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
