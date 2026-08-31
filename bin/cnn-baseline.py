import random
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torch.nn as nn
import torch
import sklearn
from torch.utils.data import TensorDataset, DataLoader

def random_seq_generator(n_seqs: int, seq_size: int):
    nucl_sequences = "ATGC"
    a = ""

    my_nucl_list = []
    for i in range(n_seqs):
        for j in range(seq_size):
            a += random.choice(nucl_sequences)
        my_nucl_list.append(a)
        a = ""

    return my_nucl_list

def plant_g4_motif(nucl_string: str, g4_motif: str):
    """
    take a nucleotide string, plant a g4 motif (predefined) inside the nucleotide string
    """

    nucl_string_len = len(nucl_string)
    g4_motif_len = len(g4_motif)
    assert nucl_string_len > g4_motif_len, "Length of nucleotide string less that the g4 motif, can't insert"

    start_index = random.choice(list(range(0,nucl_string_len-g4_motif_len + 1)))

    return nucl_string[:start_index] + g4_motif + nucl_string[start_index + g4_motif_len:]


def encode_nucl(nucl_string: str, fitted_le: LabelEncoder):

    label_encoding = fitted_le.transform(list(nucl_string)) # returns an array
    label_tensors = torch.from_numpy(label_encoding)

    ohe = F.one_hot(label_tensors, num_classes = 4).float()

    return ohe


if __name__=="__main__":
    # Start here
    # fill these up - argparse it later
    n_positive, n_negative = 960, 960
    k_mer_length = 200
    g4_motif = "GGGTTAGGGTTAGGGTTAGGG"

    # until here

    # modeling starts here, don't edit below this
    le = LabelEncoder()
    nucleotide_chars = list("ATGC")
    le.fit(nucleotide_chars)
    random_seqs_set1 = random_seq_generator(n_positive, k_mer_length)
    random_seqs_set2 = random_seq_generator(n_negative, k_mer_length)

    positive_set = []
    negative_set = []
    for seq in random_seqs_set1:
        positive_set.append(encode_nucl(plant_g4_motif(seq, g4_motif), le))

    for seq in random_seqs_set2:
        negative_set.append(encode_nucl(seq, le))


    positive_set_tensor = torch.stack(positive_set).permute(0,2,1)
    negative_set_tensor = torch.stack(negative_set).permute(0,2,1)

    print(positive_set_tensor.shape, negative_set_tensor.shape)
    # torch.Size([160, 4, 200]) torch.Size([160, 4, 200])

    # split to X and y

    X = torch.cat([positive_set_tensor, negative_set_tensor]).float()

    y = torch.cat([torch.ones(len(positive_set_tensor)), torch.zeros(len(negative_set_tensor))]).unsqueeze(1)

    print(X.shape, y.shape)
    # torch.Size([320, 4, 200]) torch.Size([320, 1])



    # shuffle and 80/20 split

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator = g)
    X, y = X[perm], y[perm]
    n_train = int(0.8 * len(X))

    X_train, X_test, y_train, y_test = X[:n_train], X[n_train:], y[:n_train], y[n_train:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size = 32, shuffle = True)

    ##

    class OneDCnnMotif(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.layer1 = nn.Conv1d(4, 16, kernel_size = kernel_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(16, 1)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = x.max(dim=2).values
            x = self.layer2(x)

            return x

    # Hyperparameters
    kernel_size = 12
    learning_rate = 0.001
    epochs = 10
    model = OneDCnnMotif(kernel_size=kernel_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # training loop


    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        acc = ((model(X_test) > 0).float() == y_test).float().mean()
    print(f"epoch {epoch+1:2d} | loss {running/n_train:.4f} | test acc {acc:.3f}")
