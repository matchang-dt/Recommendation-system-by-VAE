# %%
from collections import defaultdict
from random import seed, choice

from sklearn.metrics import accuracy_score, root_mean_squared_error

# %%
from data_pair_formatting import train_user_item_pair, test_user_item_pair, not_items_per_user, SEED

# %%
# take a portion of the train data as the validation data and generate their negative samples
train_end_idx = round(len(train_user_item_pair) * 0.95)
valid_user_item_pair = train_user_item_pair[train_end_idx:]

# %%
neg_valid_user_item_pair = []
used_items_per_user = defaultdict(set)

seed(SEED)
for user, _, _, _ in valid_user_item_pair:
    while True:
        item = choice(not_items_per_user[user])
        if item not in used_items_per_user[user]: break
    neg_valid_user_item_pair.append((user, item, -1, -1))
    used_items_per_user[user].add(item)

valid_user_item_pair = valid_user_item_pair + neg_valid_user_item_pair
del used_items_per_user

# %%
# Consider top-n% user-item interactions as popular items and recommend them.
# Count up popularity of each item and sort them.
popularity_dict = defaultdict(int)
popularity = []
total_interaction = 0

for _, item, _, _ in train_user_item_pair[:train_end_idx]:
    popularity_dict[item] += 1
    total_interaction += 1

for key, value in popularity_dict.items():
    popularity.append((value, key))

popularity.sort(key=lambda x:-x[0])

# %%
# find best threshold with validation data

ths = [round(i * 0.01, 2) for i in range(101)]

best_acc = 0
best_th = 0
accs = []
cnt = 0
for th in ths:
    sum_count = 0
    popular_set = set()
    for count, item in popularity:
        sum_count += count
        popular_set.add(item)
        if sum_count > total_interaction * th:
            break
    predicted = []
    labels = []
    for _, item, playtime, _ in valid_user_item_pair:
        if item in popular_set:
            predicted.append(1)
        else:
            predicted.append(0)
        if playtime >= 0:
            labels.append(1)
        else:
            labels.append(0)
    acc = accuracy_score(labels, predicted)
    accs.append((acc, th))
    cnt += 1
    print("progress", round(cnt / len(ths) * 100, 1), "%       ", cnt, "/", len(ths), end="     \r")
    if acc > best_acc:
        best_acc = acc
        best_th = th
accs.sort(key=lambda x:-x[0])
print("Top 5 (accuracy, threshold)")
print(accs[:5])

# %%
print("best accuracy:", best_acc, ", best threshold:", best_th)

# %%
# train with whole train data and the best threshold, and test with test data

popularity_dict = defaultdict(int)
popularity = []
total_interaction = 0

for _, item, _, _ in train_user_item_pair:
    popularity_dict[item] += 1
    total_interaction += 1

for key, value in popularity_dict.items():
    popularity.append((value, key))

popularity.sort(key=lambda x:-x[0])

# %%
sum_count = 0
popular_set = set()
for count, item in popularity:
    sum_count += count
    popular_set.add(item)
    if sum_count > total_interaction * best_th:
        break
predicted = []
labels = []
for _, item, playtime, _ in test_user_item_pair:
    if item in popular_set:
        predicted.append(1)
    else:
        predicted.append(0)
    if playtime >= 0:
        labels.append(1)
    else:
        labels.append(0)
acc = accuracy_score(labels, predicted)
print("accuracy:", acc, "threshold", best_th)

# %%
sum_count = 0
popular_set = set()
for count, item in popularity:
    sum_count += count
    popular_set.add(item)
    if sum_count > total_interaction * best_th:
        break
print("Number of popular items:", len(popular_set), "Number of all items:", len(popularity), "Ratio:", len(popular_set)/ len(popularity))
