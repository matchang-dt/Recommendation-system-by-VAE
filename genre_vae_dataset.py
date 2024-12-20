#%%
import gzip
import pandas as pd

from data_pair_formatting import user_item_pair, train_user_item_pair, test_user_item_pair, num_users, num_items, not_items_per_user, SPLITTING_RATIO, SEED

#%%
def persing(file_path):
    ret = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
        # user_set = set()
        for line in gz_file:
            new_data = eval(line)
            ret.append(eval(line))
    return ret

games = persing("steam_games.json.gz")

game_set = set()
for u, i, _, _ in user_item_pair:
    game_set.add(i)

games_aus = []
for g in games:
    if "id" not in g:
        continue
    if g["id"] in game_set:
        games_aus.append(g)

games_aus_df = pd.DataFrame(games_aus)
#%%
genres_per_item = dict()
genre_idx = {}
idx = 0
for g in games_aus:
    # print(g , end="\r")
    if "genres" in g:
        genres_per_item[g['id']] = g['genres']
        for gs in g["genres"]:
            if gs in genre_idx:
                continue
            genre_idx[gs] = idx
            idx += 1

#%%
if __name__ == "__main__":
    cnt = 0
    print(len(genres_per_item))
    for key, value in genres_per_item.items():
        print(key, value)
        cnt += 1
        if cnt > 30:
            break

