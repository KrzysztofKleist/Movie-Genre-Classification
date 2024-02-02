import argparse
import pandas as pd
import ast
import matplotlib.pyplot as plt


def get_genre_distribution(df) -> dict:
    genres_count = {}

    for genre in pd.DataFrame(df.columns).loc[3:, 0].values.tolist():
        genres_count[genre] = 0

    # Loop over subfolders in the parent folder
    for index, row in df.iterrows():
        genres_count[row["genre"]] += 1

    genres_count = dict(sorted(genres_count.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(10, 5))

    plt.bar(genres_count.keys(), genres_count.values())

    plt.xlabel("Genres")
    plt.ylabel("Occurrence")
    plt.title("Occurence of genres")
    # Choose one of 2 below
    # plt.savefig("files_slcr/data_distribution_data_train_random_aug_all.png")
    # plt.savefig("files_slcr/data_distribution_data_train_separate_movies_aug_all.png")
    
    plt.show()

    return genres_count


if __name__ == "__main__":
    # Choose one of 2 below
    df = pd.read_csv("files_slcr/data_train_random_aug_all.tsv", sep="\t")
    # df = pd.read_csv("files_slcr/data_train_separate_movies_aug_all.tsv", sep="\t")
    genres = get_genre_distribution(df)
    print(genres)
