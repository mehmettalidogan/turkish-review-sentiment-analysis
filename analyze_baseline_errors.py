"""
Baseline TF-IDF + Logistic Regression hatalarını inceler.

Kayıtlı modeli ve vektörleştiriciyi yalnızca okur; yeniden eğitmez ve üzerine yazmaz.
Test tahminleri, eğitimde hiç kullanılmayan dataset/test_sentiment.csv üzerinden üretilir.
"""

from pathlib import Path
import re

import joblib
import pandas as pd

TEST_PATH = Path("dataset/test_sentiment.csv")
VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")
CLASSIFIER_PATH = Path("models/logreg_sentiment.joblib")
ERROR_PATH = Path("analysis/baseline_misclassified_test.csv")

CLASS_ORDER = ["Negative", "Neutral", "Positive"]
PAIR_ORDER = [
    ("Neutral", "Negative"),
    ("Neutral", "Positive"),
    ("Negative", "Neutral"),
    ("Positive", "Neutral"),
    ("Negative", "Positive"),
    ("Positive", "Negative"),
]

# Modelin gördüğü kelimelerle aynı sınır: en az iki Unicode sözcük karakteri.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Duygu ipuçları silinmez; yalnızca hatalı yorumlarda ne kadar sık olduklarını sayarız.
PATTERNS = {
    "ama / fakat / ancak": re.compile(r"\b(?:ama|fakat|lakin|ancak)\b", re.IGNORECASE),
    "değil": re.compile(r"\bde[gğ]il\b", re.IGNORECASE),
    "yok": re.compile(r"\byok(?:tur|tu)?\b", re.IGNORECASE),
    "hiç": re.compile(r"\bhi[cç]\b", re.IGNORECASE),
    "iade": re.compile(r"\biade\w*\b", re.IGNORECASE),
    "beden / kalıp": re.compile(r"\b(?:beden\w*|kal[ıi]p|dar|bol|k[uü]c[uü]k|b[uü]y[uü]k)\b", re.IGNORECASE),
    "övgü (güzel, iyi, memnun)": re.compile(r"\b(?:g[uü]zel|iyi|memnun\w*|be[gğ]en\w*|kaliteli)\b", re.IGNORECASE),
    "şikâyet (kötü, bozuk, pişman)": re.compile(
        r"\b(?:k[oö]t[uü]|berbat|rezalet|bozuk|pi[sş]man\w*|dandik|y[ıi]rt[ıi]k|kusurlu|defolu)\b",
        re.IGNORECASE,
    ),
    "Devamını Oku": re.compile(r"devam[ıi]n[ıi]\s+oku", re.IGNORECASE),
}


def load_test() -> pd.DataFrame:
    df = pd.read_csv(TEST_PATH)
    if df["comment"].isna().any() or df["sentiment"].isna().any():
        raise ValueError("Test kümesinde boş comment veya sentiment var.")
    return df


def tokens_of(text: str) -> list[str]:
    """Baseline vektörleştirici gibi küçük harfe çevirip kelime ayırır."""
    return TOKEN_RE.findall(str(text).lower())


def pattern_rate(texts: pd.Series, pattern: re.Pattern) -> float:
    if len(texts) == 0:
        return 0.0
    return float(texts.astype(str).str.contains(pattern, regex=True).mean())


def short_rate(texts: pd.Series) -> float:
    """Üç veya daha az kelimesi olan yorumların payı."""
    if len(texts) == 0:
        return 0.0
    counts = texts.astype(str).map(lambda text: len(tokens_of(text)))
    return float((counts <= 3).mean())


def contrasting_tokens(error_texts: pd.Series, correct_texts: pd.Series, limit: int = 12) -> pd.DataFrame:
    """
    Hatalı grupta, aynı gerçek sınıftaki doğru tahminlere göre daha sık geçen kelimeler.
    Tek başına sık olan 'ürün' gibi kelimeler, doğru grupta da sıksa listede yükselmez.
    """
    def document_counts(texts: pd.Series) -> dict[str, int]:
        counts: dict[str, int] = {}
        for text in texts.astype(str):
            for token in set(tokens_of(text)):
                counts[token] = counts.get(token, 0) + 1
        return counts

    error_counts = document_counts(error_texts)
    correct_counts = document_counts(correct_texts)
    n_error = max(len(error_texts), 1)
    n_correct = max(len(correct_texts), 1)
    rows = []
    for token, error_df in error_counts.items():
        if error_df < 15:
            continue
        correct_df = correct_counts.get(token, 0)
        error_share = error_df / n_error
        correct_share = correct_df / n_correct
        rows.append(
            {
                "kelime": token,
                "hatalı_yorum": error_df,
                "hatalı_pay": error_share,
                "doğru_pay": correct_share,
                "fark": error_share - correct_share,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["kelime", "hatalı_yorum", "hatalı_pay", "doğru_pay", "fark"])
    table = pd.DataFrame(rows).sort_values(["fark", "hatalı_yorum"], ascending=False)
    return table.head(limit)


def main() -> None:
    test_df = load_test()
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)

    # Teste yalnızca transform. Sözlük eğitimde kilitlendi; burada fit yok.
    x_test = vectorizer.transform(test_df["comment"].astype(str))
    predicted = classifier.predict(x_test)

    examined = test_df.copy()
    examined["true_sentiment"] = examined["sentiment"].astype(str)
    examined["predicted_sentiment"] = predicted
    examined["is_correct"] = examined["true_sentiment"] == examined["predicted_sentiment"]

    # Aynı yorum metni test içinde birden fazla duygu etiketi taşıyorsa model birini seçmek zorunda.
    label_counts = examined.groupby("comment")["true_sentiment"].nunique()
    conflict_comments = set(label_counts[label_counts > 1].index)
    examined["label_conflict"] = examined["comment"].isin(conflict_comments)

    errors = examined.loc[~examined["is_correct"]].copy()
    errors["error_pair"] = errors["true_sentiment"] + " → " + errors["predicted_sentiment"]

    ERROR_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERROR_PATH, index=False, encoding="utf-8-sig")

    print("=== Hata özeti ===")
    print(f"Test: {len(examined)} | Doğru: {int(examined['is_correct'].sum())} | Yanlış: {len(errors)}")
    print(f"Kayıt: {ERROR_PATH}")
    print("\nGerçek → tahmin")
    pair_counts = (
        errors.groupby(["true_sentiment", "predicted_sentiment"]).size().reindex(PAIR_ORDER, fill_value=0)
    )
    for (true_label, pred_label), count in pair_counts.items():
        true_total = int((examined["true_sentiment"] == true_label).sum())
        print(f"  {true_label} → {pred_label}: {int(count)} ({count / true_total:.1%} of {true_label})")

    conflict_errors = int(errors["label_conflict"].sum())
    print("\n=== Etiket çatışması ===")
    print(f"Testte birden fazla sentiment taşıyan farklı yorum: {len(conflict_comments)}")
    print(f"Yanlış satırların {conflict_errors} tanesi ({conflict_errors / len(errors):.1%}) bu yorumlarda.")
    print("Bu satırlarda metin tek başına bir sınıfa ait değil; yıldız satırlar arasında değişiyor.")

    print("\n=== Örüntü oranları ===")
    print("Pay, o gruptaki yorumların yüzde kaçı örüntüyü içeriyor.")
    groups = {"doğru": examined.loc[examined["is_correct"], "comment"]}
    for true_label, pred_label in PAIR_ORDER:
        mask = (errors["true_sentiment"] == true_label) & (errors["predicted_sentiment"] == pred_label)
        groups[f"{true_label}→{pred_label}"] = errors.loc[mask, "comment"]

    header = "örüntü".ljust(32) + "".join(name.rjust(18) for name in ["doğru", "Neu→Neg", "Neu→Pos", "Neg→Neu", "Pos→Neu"])
    print(header)
    focus = ["doğru", "Neutral→Negative", "Neutral→Positive", "Negative→Neutral", "Positive→Neutral"]
    for name, pattern in PATTERNS.items():
        cells = [f"{pattern_rate(groups[key], pattern):.1%}".rjust(18) for key in focus]
        print(name.ljust(32) + "".join(cells))
    short_cells = [f"{short_rate(groups[key]):.1%}".rjust(18) for key in focus]
    print("en fazla 3 kelime".ljust(32) + "".join(short_cells))

    correct = examined.loc[examined["is_correct"]]
    print("\n=== Hatalarda ayrıştırıcı kelimeler ===")
    print("Aynı gerçek sınıftaki doğru tahminlere göre payı daha yüksek kelimeler. Eşik: en az 15 hatalı yorum.")
    for true_label, pred_label in PAIR_ORDER[:4]:
        error_texts = errors.loc[
            (errors["true_sentiment"] == true_label) & (errors["predicted_sentiment"] == pred_label),
            "comment",
        ]
        correct_texts = correct.loc[correct["true_sentiment"] == true_label, "comment"]
        table = contrasting_tokens(error_texts, correct_texts)
        print(f"\n{true_label} → {pred_label} (n={len(error_texts)})")
        if table.empty:
            print("  Eşik üstünde kelime yok.")
            continue
        for row in table.itertuples(index=False):
            print(
                f"  {row.kelime:<16} hatalı {row.hatalı_pay:6.1%} | doğru {row.doğru_pay:6.1%} | fark {row.fark:+6.1%}"
            )

    # Altı karışımın her birinden 5 örnek: toplam 30, tohum sabit.
    print("\n=== 30 yanlış tahmin ===")
    shown = 0
    for true_label, pred_label in PAIR_ORDER:
        part = errors.loc[
            (errors["true_sentiment"] == true_label) & (errors["predicted_sentiment"] == pred_label)
        ]
        sample = part.sample(n=5, random_state=42)
        print(f"\n--- {true_label} → {pred_label} ---")
        for row in sample.itertuples(index=False):
            shown += 1
            comment = str(row.comment).replace("\n", " ")
            if len(comment) > 200:
                comment = comment[:197] + "..."
            conflict = " | etiket çatışması" if row.label_conflict else ""
            print(f"{shown}. gerçek: {row.true_sentiment} | tahmin: {row.predicted_sentiment}{conflict}")
            print(f"   {comment}")
    if shown < 30:
        raise RuntimeError(f"En az 30 örnek bekleniyordu, {shown} gösterildi.")


if __name__ == "__main__":
    main()
