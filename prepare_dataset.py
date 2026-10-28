import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ORIGINAL_PATH = "dataset/trendyol_reviews_20251212_000835.csv"
BALANCED_PATH = "dataset/balanced_sentiment.csv"
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 9000

POINT_TO_SENTIMENT = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive",
}


def main():
    # Orijinal dosya sadece okunur. Tüm işlemler bellek kopyası üzerinde yapılır.
    df = pd.read_csv(ORIGINAL_PATH)
    print("Orijinal dosya okundu, uzerine yazilmayacak.")
    print(f"Kaynak satir sayisi: {len(df)}")

    df["sentiment"] = df["point"].map(POINT_TO_SENTIMENT)
    unmapped = df["sentiment"].isna().sum()
    print(f"Eslesmeyen point degeri: {unmapped}")

    print("\n[1] Bos comment kontrolu")
    null_count = df["comment"].isna().sum()
    empty_count = df["comment"].fillna("").astype(str).str.strip().eq("").sum()
    print(f"Null comment: {null_count}")
    print(f"Bos veya sadece bosluk olan comment: {empty_count}")

    print("\n[2] Sentiment dagilimi (tum veri)")
    print(df["sentiment"].value_counts())

    print("\n[3] Ayni comment, farkli sentiment")
    sentiments_per_comment = df.groupby("comment")["sentiment"].nunique()
    conflict_count = int((sentiments_per_comment > 1).sum())
    print(f"Birden fazla sentiment sinifinda gecen benzersiz comment: {conflict_count}")

    if conflict_count > 0:
        conflict_texts = sentiments_per_comment[sentiments_per_comment > 1].index
        conflict_rows = df[df["comment"].isin(conflict_texts)]
        pair_counts = (
            conflict_rows.groupby("comment")["sentiment"]
            .agg(lambda values: " + ".join(sorted(set(values))))
            .value_counts()
        )
        print("Catisan etiket kombinasyonlari:")
        print(pair_counts)

    print("\n[4] Dengeli calisma dataseti")
    parts = []
    for label in ["Negative", "Neutral", "Positive"]:
        subset = df[df["sentiment"] == label]
        if len(subset) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"{label} sinifinda {len(subset)} satir var, {SAMPLES_PER_CLASS} secilemez."
            )
        parts.append(subset.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_STATE))

    balanced = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    print("\n[5] Dengeli sette duplicate comment")
    total_rows = len(balanced)
    unique_comments = balanced["comment"].nunique()
    extra_duplicate_rows = total_rows - unique_comments
    repeated_comments = int(balanced["comment"].duplicated(keep=False).sum())
    print(f"Toplam satir: {total_rows}")
    print(f"Benzersiz comment: {unique_comments}")
    print(f"Fazladan tekrar satiri (toplam - benzersiz): {extra_duplicate_rows}")
    print(f"Tekrar eden bir yoruma ait satir sayisi: {repeated_comments}")
    print("Duplicate yorumlar silinmedi.")

    balanced_conflicts = (
        balanced.groupby("comment")["sentiment"].nunique().gt(1).sum()
    )
    print(f"Dengeli sette farkli sentiment'e dusen comment: {int(balanced_conflicts)}")

    balanced.to_csv(BALANCED_PATH, index=False)
    print(f"\nKaydedildi: {BALANCED_PATH}")

    print("\n[6] Son dataset ozeti")
    print(f"Toplam satir: {len(balanced)}")
    print(f"Sutunlar: {list(balanced.columns)}")
    print("Sentiment dagilimi:")
    print(balanced["sentiment"].value_counts())
    print(f"Benzersiz comment: {unique_comments}")
    print(f"Duplicate comment sayisi (fazla satir): {extra_duplicate_rows}")

    print("\nHer siniftan 3 ornek yorum:")
    for label in ["Negative", "Neutral", "Positive"]:
        print(f"\n--- {label} ---")
        examples = balanced[balanced["sentiment"] == label]["comment"].head(3)
        for index, text in enumerate(examples, start=1):
            print(f"{index}. {text}")


if __name__ == "__main__":
    main()
