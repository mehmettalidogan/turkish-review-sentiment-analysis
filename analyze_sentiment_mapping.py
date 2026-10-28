import pandas as pd
import numpy as np

def analyze_sentiment():
    file_path = "dataset/trendyol_reviews_20251212_000835.csv"
    print(f"--- Analyzing Sentiment Mapping for {file_path} ---")
    
    df = pd.read_csv(file_path)
    
    # 5. Unique comments count
    total_comments = len(df)
    unique_comments = df['comment'].nunique()
    print(f"\n[1] Comment Duplication:")
    print(f"Total Comments: {total_comments}")
    print(f"Unique Comments: {unique_comments}")
    print(f"Duplicated Comment Count: {total_comments - unique_comments}")
    print(f"Most Frequent Comments:")
    print(df['comment'].value_counts().head(5))

    # 2. Average comment length per star rating
    print("\n[2] Average Comment Length per Star Rating:")
    # Ensure comments are strings
    df['comment_str'] = df['comment'].astype(str)
    df['comment_len'] = df['comment_str'].apply(len)
    avg_len = df.groupby('point')['comment_len'].mean().sort_index()
    print(avg_len)
    
    # 3. Empty or meaningless comments (length <= 3)
    print("\n[3] Meaningless/Very Short Comments (Length <= 3) per Star Rating:")
    short_comments = df[df['comment_len'] <= 3]
    short_counts = short_comments.groupby('point').size().reindex([1,2,3,4,5], fill_value=0)
    print(short_counts)
    
    print("\n[4] Sample 10 Random Comments per Star Rating:")
    # Set seed for reproducibility
    np.random.seed(42)
    
    for p in [1, 2, 3, 4, 5]:
        print(f"\n--- {p} Yıldız Yorumları ---")
        subset = df[df['point'] == p]
        
        if len(subset) >= 10:
            sample = subset.sample(10)['comment_str'].tolist()
        else:
            sample = subset['comment_str'].tolist()
            
        for i, text in enumerate(sample, 1):
            # Truncate if too long for readability
            display_text = text if len(text) < 150 else text[:147] + "..."
            print(f"{i}. {display_text}")

if __name__ == "__main__":
    analyze_sentiment()
