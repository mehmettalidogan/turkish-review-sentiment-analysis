import pandas as pd
import os

def analyze_dataset(file_path):
    print(f"--- Analyzing Dataset: {file_path} ---")
    
    # Check file size
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")
    
    # Load dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read CSV, attempting with different encoding or separator. Error: {e}")
        try:
            df = pd.read_csv(file_path, sep=';')
        except:
            print("Failed again. Exiting.")
            return

    # 1. Row Count & Column Names
    print(f"\n[1] Row Count: {len(df)}")
    print(f"[2] Columns: {list(df.columns)}")
    
    # 2. Data Types
    print("\n[3] Data Types:")
    print(df.dtypes)
    
    # 3. Missing Values
    print("\n[4] Missing Values:")
    print(df.isnull().sum())
    
    # 4. Duplicates
    print(f"\n[5] Duplicate Rows Count: {df.duplicated().sum()}")
    
    # 5. First 5 Rows
    print("\n[6] First 5 Rows:")
    print(df.head())
    
    # 6. Distribution of Potential Target/Source Columns
    print("\n[7] Distribution of Categorical/Target Columns:")
    keywords = ['rating', 'star', 'score', 'label', 'sentiment', 'source', 'puan', 'kaynak', 'duygu']
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in keywords) or df[col].dtype == 'object' and df[col].nunique() < 20:
            print(f"--- Column: {col} ---")
            print(df[col].value_counts(dropna=False))
            print("-" * 20)

    # 7. Identify Text and Target columns (Heuristics)
    print("\n[8] Automated Identification:")
    text_col = None
    max_len = 0
    for col in df.select_dtypes(include=['object']):
        # Find column with longest average string length -> likely review text
        avg_len = df[col].dropna().astype(str).str.len().mean()
        if avg_len > max_len:
            max_len = avg_len
            text_col = col
            
    if text_col:
        print(f"-> Most likely 'Review Text' column: '{text_col}' (Average length: {max_len:.2f} chars)")
        
if __name__ == "__main__":
    analyze_dataset("dataset/trendyol_reviews_20251212_000835.csv")
