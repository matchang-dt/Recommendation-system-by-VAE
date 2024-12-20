# %%
# This model simply determines recommendation or not by the threshold.
# Although the output is inferior to the top N recommendation, the processing time is much faster. 
# %%
from collections import defaultdict
from math import ceil
from copy import deepcopy
from random import choice, seed, sample
from sklearn.metrics import accuracy_score, root_mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from datetime import datetime

# %%
from genre_vae_dataset import genres_per_item, genre_idx, train_users_items_pair, test_users_items_pair, not_items_per_user, num_users, num_items, SPLITTING_RATIO, SEED
from cse258model.vae import GenreVAE, loss_function4binary, loss_function4regression

# %%
hyperparams = {
    "mode": "binary", # "binary" or "regression"
    "learning_rate": 0.001,
    "batch_size": 64,
    "input_dim": num_items,
    "hidden_dim": 256,
    "latent_dim": 50,
    "epochs": 20,
    "lambda": 0,
    "noise_ratio": 0.5
}

NUM_GENRE = len(genre_idx)

# %%
user_idx = dict()
item_idx = dict()

uid, iid = 0, 0
for user, item, _, _ in train_users_items_pair:
    if user not in user_idx:
        user_idx[user] = uid
        uid += 1
    if item not in item_idx:
        item_idx[item] = iid
        iid += 1

for user, item, _, _ in test_users_items_pair:
    if user not in user_idx:
        user_idx[user] = uid
        uid += 1
    if item not in item_idx:
        item_idx[item] = iid
        iid += 1

# %%
train_end_idx = round(len(train_users_items_pair) * 0.95)
valid_users_items_pair = train_users_items_pair[train_end_idx:]

# %%
neg_valid_users_items_pair = []
used_items_per_user = defaultdict(set)

seed(SEED)
for user, _, _, _ in valid_users_items_pair:
    while True:
        item = choice(not_items_per_user[user])
        if item not in used_items_per_user[user]: break
    neg_valid_users_items_pair.append((user, item, -1, -1))
    used_items_per_user[user].add(item)

valid_users_items_pair = valid_users_items_pair + neg_valid_users_items_pair
del used_items_per_user, not_items_per_user

# %%
class DAEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx

# %%
train_mat = torch.zeros(num_users, num_items, dtype=torch.float32)

for user, item, _, _ in train_users_items_pair[:train_end_idx]:
    train_mat[user_idx[user], item_idx[item]] = 1.

nonzero_idx = []
for row in train_mat:
    nonzero_idx.append(list(torch.nonzero(row, as_tuple=True)[0]))

train_loader = DataLoader(DAEDataset(train_mat), batch_size=hyperparams["batch_size"], shuffle=True)

# %%
genre_train_mat = torch.zeros(num_users, NUM_GENRE, dtype=torch.float32)

for user, item, _, _ in train_users_items_pair:
    if item not in genres_per_item:
        continue
    gs = genres_per_item[item]
    for g in gs:
        genre_train_mat[user_idx[user], genre_idx[g]] += 1.

for i in range(len(genre_train_mat)):
    row = genre_train_mat[i]
    if row.sum() == 0:
        continue
    row = row / row.sum()
    genre_train_mat[i] = row

# %%
torch.save(train_mat, "train_mat.pt")

# %%
vae = GenreVAE(hyperparams)
optimizer = optim.Adam(vae.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["lambda"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)

# %%
def add_noise(batch, nonzero_idcs, noise_ratio):
    new_batch = deepcopy(batch)
    for idcs, row in zip(nonzero_idcs, new_batch):
        cols = torch.tensor(sample(idcs, ceil(len(idcs) * noise_ratio)), dtype=torch.int32)
        row[cols] = 0
    return new_batch

# %%
seed(SEED)
for epoch in range(hyperparams["epochs"]):
    loop_len = len(train_mat)
    cur = 0
    for batch, idcs in train_loader:
        print(round(cur / loop_len * 100, 1), end=" %\r")
        cur += batch.shape[0]
        batch = batch.to(device)
        optimizer.zero_grad()
        nonzero_idcs = [nonzero_idx[i] for i in idcs] 
        noised_batch = add_noise(batch, nonzero_idcs, hyperparams['noise_ratio'])
        noised_batch.to(device)
        genre_batch = genre_train_mat[idcs]
        recon_batch, mu, logvar = vae(noised_batch, genre_batch)
        loss = loss_function4binary(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# %%
vae.eval()
with torch.no_grad():    
    user_latent = vae.encoder(train_mat)
    genre_embeded = vae.embeder(genre_train_mat)
    mu = vae.mu(torch.cat([user_latent, genre_embeded], dim=1))
    z = mu
    recon_mat = vae.decoder(z)
del train_mat

# %%
# ths = [round(i * 0.1, 1) for i in range(10)]
ths = [round(i * 0.01, 2) for i in range(100)]
# ths = [round(i * 0.001, 3) for i in range(10)]
# ths = [round(i * 0.0001, 4) for i in range(50, 61)]
accs = []
for th in ths:
    predicted = []
    labels = []
    for user, item, playtime, _ in valid_users_items_pair:
        predicted.append(int(recon_mat[user_idx[user], item_idx[item]] > th))
        labels.append(int((playtime >= 0)))
    accs.append((accuracy_score(labels, predicted), th))
accs.sort(key=lambda x:-x[0])
print("accuracy, threshold")
print(accs[:3])
best_th = accs[0][1]

# %%
train_mat = torch.zeros(num_users, num_items, dtype=torch.float32)

for user, item, _, _ in train_users_items_pair:
    train_mat[user_idx[user], item_idx[item]] = 1.

train_loader = DataLoader(TensorDataset(train_mat), batch_size=hyperparams["batch_size"], shuffle=True)

# %%
vae = GenreVAE(hyperparams)
optimizer = optim.Adam(vae.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["lambda"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)

# %%
for epoch in range(hyperparams["epochs"]):
    for batch in train_loader:
        batch = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(batch)
        loss = loss_function4binary(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')


# %%
now = datetime.now().strftime("%y%m%d%H%M%S")

epochs = hyperparams["epochs"]
noise = hyperparams["noise_ratio"]
torch.save(vae.state_dict(), f"trained_model/vae_test_e{epochs}_n{int(noise * 100)}_" + now +".pth")
print("model saved!")


# %%
vae.eval()
with torch.no_grad():    
    user_latent = vae.encoder(train_mat)
    mu = vae.mu(user_latent)
    z = mu
    recon_mat = vae.decoder(z)
del train_mat

# %%
predicted = []
labels = []

for user, item, playtime, _ in test_users_items_pair:
    predicted.append(int(recon_mat[user_idx[user], item_idx[item]] > best_th))
    labels.append(int((playtime >= 0)))

# %%
print(accuracy_score(labels, predicted))
