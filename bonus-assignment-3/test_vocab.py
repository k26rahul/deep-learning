import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from collections import Counter
import ast

def build_vocab_1(texts, min_freq=2):
    all_words = [word for text in texts for word in text]
    word_counts = Counter(all_words)
    
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for word, count in word_counts.items():
        if count >= min_freq and len(vocab) < 10000:
            vocab[word] = len(vocab)
    return vocab

def build_vocab_2(texts, min_freq=2):
    all_words = [word for text in texts for word in text]
    word_counts = Counter(all_words)
    
    vocab = {'<UNK>': 0, '<PAD>': 1}
    for word, count in word_counts.most_common():
        if count >= min_freq and len(vocab) < 10000:
            vocab[word] = len(vocab)
    return vocab

def main():
    torch.manual_seed(42)
    file_path = "c:\\k26rahul\\Code\\Work\\deep-learning\\preprocessed_yelp_data.csv"
    df = pd.read_csv(file_path)
    df['text'] = df['text'].apply(ast.literal_eval)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    train_texts = train_df['text'].tolist()
    
    vocab1 = build_vocab_1(train_texts)
    vocab2 = build_vocab_2(train_texts)
    
    print(f"Vocab 1 size: {len(vocab1)}")
    print(f"Vocab 2 size: {len(vocab2)}")
    
    first_text = train_texts[0]
    
    num1 = [vocab1.get(word, 0) for word in first_text]
    num2 = [vocab2.get(word, 0) for word in first_text]
    
    print(f"First numericalized (vocab1 - insertion order): {num1[:15]}")
    print(f"First numericalized (vocab2 - frequency order): {num2[:15]}")

if __name__ == "__main__":
    main()
