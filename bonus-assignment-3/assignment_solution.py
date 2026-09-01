import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import ast
from collections import Counter

# Set torch.manual_seed to be 42
seed = 42
torch.manual_seed(seed)

def build_vocab(texts, max_size=10000, min_freq=2):
    all_words = [word for text in texts for word in text]
    word_counts = Counter(all_words)
    
    vocab = {'<UNK>': 0, '<PAD>': 1}
    # Insertion order as per problem description matching Option D in Q5
    for word, count in word_counts.items():
        if count >= min_freq and len(vocab) < max_size:
            vocab[word] = len(vocab)
            
    return vocab

def numericalize_text(text, vocab):
    return [vocab.get(word, 0) for word in text]

class YelpDataset(Dataset):
    def __init__(self, dataframe, max_seq_length):
        self.dataframe = dataframe
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['numericalized_text']
        label = self.dataframe.iloc[idx]['label']
        
        if len(text) < self.max_seq_length:
            text = text + [1] * (self.max_seq_length - len(text))
        else:
            text = text[:self.max_seq_length]
            
        return {'text': torch.tensor(text), 'label': torch.tensor(label, dtype=torch.float32)}

class RNNModel(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=1)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        out = self.linear(hidden[-1])
        return out.squeeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    torch.manual_seed(42)
    file_path = "c:\\k26rahul\\Code\\Work\\deep-learning\\preprocessed_yelp_data.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    df['text'] = df['text'].apply(ast.literal_eval)

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    train_texts = train_df['text'].tolist()
    
    vocab = build_vocab(train_texts, max_size=10000, min_freq=2)
    print(f"Q3: Size of vocabulary: {len(vocab)}")
    
    train_df['numericalized_text'] = train_df['text'].apply(lambda x: numericalize_text(x, vocab))
    test_df['numericalized_text'] = test_df['text'].apply(lambda x: numericalize_text(x, vocab))
    
    first_numericalized = train_df.iloc[0]['numericalized_text']
    print(f"Q4: First element after numericalizing: {first_numericalized}")
    
    max_seq_length = 100
    train_dataset = YelpDataset(train_df, max_seq_length)
    test_dataset = YelpDataset(test_df, max_seq_length)
    
    first_tensor_element = train_dataset[0]
    print(f"Q5: First element after tensor conversion: {first_tensor_element}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_size = len(vocab)
    embedding_dim = 100
    hidden_size = 256
    num_layers = 2
    output_size = 1
    
    model = RNNModel(input_size, embedding_dim, hidden_size, num_layers, output_size)
    
    print(f"Q6: Total parameters in RNN model: {count_parameters(model)}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 2
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            text = batch['text']
            label = batch['label']
            
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
    print(f"Q7: Training Loss after 2 epochs: {total_loss/len(train_loader):.4f}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            text = batch['text']
            label = batch['label']
            
            predictions = model(text)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct += (rounded_preds == label).sum().item()
            total += label.size(0)
            
    accuracy = correct / total * 100
    print(f"Q8: Test Accuracy after 2 epochs: {accuracy:.2f}")

if __name__ == "__main__":
    main()
