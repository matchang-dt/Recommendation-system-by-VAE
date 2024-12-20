# %%
from collections import defaultdict
import gzip
from  random import seed, shuffle, choice
from time import time

# %%
SEED = 42
SPLITTING_RATIO = 0.95

# %%
def gen_user_item_pair(file_path):
    """
    Read the original file and return user-item interaction list
    """
    user_item_pair = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        user_set = set()
        for line in gz_file:
            d = eval(line)
            if d['steam_id'] in user_set:
                continue
            user_set.add(d['steam_id'])
            for it in d['items']:
                user_item_pair.append((d['steam_id'], it['item_id'], it['playtime_forever'], it['playtime_2weeks']))
    seed(SEED)
    shuffle(user_item_pair)
    
    return user_item_pair

# %%
start = time()
print("reading the file and generating user-item interaction pair ... (would take 1 or 2 minutes)", end="")
user_item_pair = gen_user_item_pair("australian_users_items.json.gz")
print("   time:", round(time() - start), "s" )

# %%
train_last_idx = int(len(user_item_pair) * SPLITTING_RATIO)
train_user_item_pair = user_item_pair[:train_last_idx]
test_user_item_pair = user_item_pair[train_last_idx:]

# %%
items_per_user = defaultdict(set)
not_items_per_user = dict() # item set each user does NOT have
all_item_set = set()

start = time()
print("generating negative samples... (would take 1 or 2 minutes)", end="")
for user, item, _, _ in user_item_pair:
    all_item_set.add(item)
    items_per_user[user].add(item)

if __name__ != "__main__":
    for user, items in items_per_user.items():
        # covert to list because choice from set takes a long time
        not_items_per_user[user] = list(all_item_set - items_per_user[user])

del items_per_user

# %%
neg_test_user_item_pair = []
used_items_per_user = defaultdict(set)

seed(SEED)
for user, _, _, _ in test_user_item_pair:
    while True:
        item = choice(not_items_per_user[user])
        if item not in used_items_per_user[user]: break
    neg_test_user_item_pair.append((user, item, -1, -1))
    used_items_per_user[user].add(item)
del used_items_per_user

# %%
test_user_item_pair = test_user_item_pair + neg_test_user_item_pair
print("   time:", round(time() - start), "s" )

num_users = len(not_items_per_user)
num_items = len(all_item_set)

del all_item_set

# %%
if __name__ == "__main__":
    print("nun_interactions:", len(user_item_pair))
    print("num_users:", num_users)
    print("num_items:", num_items)

# %%
if __name__ == "__main__":    
    from math import log2
    import matplotlib.pyplot as plt
    
    logged_playtime = []
    for _, _, playtime, _ in user_item_pair:
        if playtime > 0:
            logged_playtime.append(log2(playtime+1))

    print("number of interactions without zero play time:", len(logged_playtime))
    logged_playtime.sort()
    logged_playtime = logged_playtime[:round(len(logged_playtime)*0.99)]
    print(logged_playtime[0], logged_playtime[-1])
    plt.hist(logged_playtime, bins=28, range=(1,15))
    plt.show()


    