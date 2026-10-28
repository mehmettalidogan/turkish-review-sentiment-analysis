import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# Kaynak dosya sadece okunur. Train ve test ayrı dosyalara yazılır.
BALANCED_PATH = "dataset/balanced_sentiment.csv"
TRAIN_PATH = "dataset/train_sentiment.csv"
TEST_PATH = "dataset/test_sentiment.csv"

RANDOM_STATE = 42
TEST_RATIO = 0.20
CLASSES = ["Negative", "Neutral", "Positive"]


def main():
    df = pd.read_csv(BALANCED_PATH)
    print(f"Okunan satir: {len(df)}")
    print("balanced_sentiment.csv degistirilmeyecek.")

    # Her benzersiz yorum bir gruptur. Grubun sinif sayilari birlikte tasinir.
    group_counts = pd.crosstab(df["comment"], df["sentiment"]).reindex(
        columns=CLASSES, fill_value=0
    )
    group_counts["n_rows"] = group_counts[CLASSES].sum(axis=1)
    group_counts["n_labels"] = (group_counts[CLASSES] > 0).sum(axis=1)

    conflict_groups = group_counts[group_counts["n_labels"] > 1]
    print("\n[1] Farkli sentiment etiketi tasiyan yorumlar")
    print(f"Benzersiz comment: {len(conflict_groups)}")
    print(f"Satir sayisi: {int(conflict_groups['n_rows'].sum())}")
    print("Bu yorumlar silinmedi ve etiketleri degistirilmedi.")
    print("Ayni yorumun tum satirlari ya egitime ya teste gidecek.")

    class_totals = df["sentiment"].value_counts().reindex(CLASSES)
    # Her sinifin test hedefi, o sinifin satir sayisinin %20'sidir.
    # 9000 * 0.20 = 1800 oldugu icin hedefler tam sayidir.
    targets = (class_totals * TEST_RATIO).round().astype(int)
    print("\n[2] Test hedefleri (sinif basina satir)")
    print(targets.to_string())

    # Gruplari karistir. Seed sabit oldugu icin ayni atama tekrar uretilir.
    shuffled = group_counts.sample(frac=1, random_state=RANDOM_STATE)

    test_counts = {label: 0 for label in CLASSES}
    test_comments = []

    for comment, row in shuffled.iterrows():
        # Grubu teste eklemek herhangi bir sinifin hedefini asiyorsa egitime birak.
        # Boylece test, sinif basina %20'yi gecmez; gruplar da bolunmez.
        fits = all(
            test_counts[label] + int(row[label]) <= int(targets[label])
            for label in CLASSES
        )
        if fits:
            test_comments.append(comment)
            for label in CLASSES:
                test_counts[label] += int(row[label])

    test_comment_set = set(test_comments)
    is_test = df["comment"].isin(test_comment_set)
    test_df = df.loc[is_test].copy()
    train_df = df.loc[~is_test].copy()

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\n[3] Sonuc")
    print(f"Egitim satiri: {len(train_df)}")
    print(f"Test satiri: {len(test_df)}")
    print(f"Toplam: {len(train_df) + len(test_df)}")
    print(f"\nKaydedildi: {TRAIN_PATH}")
    print(f"Kaydedildi: {TEST_PATH}")

    print("\nEgitim sinif dagilimi:")
    print(train_df["sentiment"].value_counts().reindex(CLASSES))
    print("\nTest sinif dagilimi:")
    print(test_df["sentiment"].value_counts().reindex(CLASSES))

    shared = set(train_df["comment"]).intersection(test_df["comment"])
    print(f"\nOrtak comment sayisi: {len(shared)}")

    # Dosyalar yazildiktan sonra ayni kontrolu diskten de yap.
    train_saved = pd.read_csv(TRAIN_PATH)
    test_saved = pd.read_csv(TEST_PATH)
    shared_saved = set(train_saved["comment"]).intersection(test_saved["comment"])
    print(f"Dosyadan okunan ortak comment sayisi: {len(shared_saved)}")
    print(f"Dosya satirlari: egitim {len(train_saved)}, test {len(test_saved)}")


if __name__ == "__main__":
    main()
