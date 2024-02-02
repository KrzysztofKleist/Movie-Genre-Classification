import argparse
import pandas as pd
import ast
import matplotlib.pyplot as plt


def get_genre_distribution(df) -> dict:
    existing_genres = {}

    # Loop over subfolders in the parent folder
    for index, row in df.iterrows():
        genre = "_".join(ast.literal_eval(row["genre"]))

        if genre not in existing_genres:
            existing_genres[genre] = 1
        else:
            existing_genres[genre] += 1

    existing_genres = dict(
        sorted(existing_genres.items(), key=lambda x: x[1], reverse=True)
    )

    single_genre_list = []

    for gen in existing_genres.keys():
        gen_list = gen.split("_")

        for g in gen_list:
            if g not in single_genre_list:
                single_genre_list.append(g)

    existing_single_genres = {}

    for gen in single_genre_list:
        if gen not in existing_single_genres:
            existing_single_genres[gen] = 0

        for g in existing_genres:
            if gen in g:
                existing_single_genres[gen] += existing_genres[g]

    existing_single_genres = dict(
        sorted(existing_single_genres.items(), key=lambda x: x[1], reverse=True)
    )

    plt.figure(figsize=(10, 5))

    plt.bar(existing_single_genres.keys(), existing_single_genres.values())

    plt.xlabel("Genres")
    plt.ylabel("Occurrence")
    plt.title("Occurence of genres")
    # Choose one of 4 below
    # plt.savefig("files/data_distribution_data_train_random_aug_all.png")
    # plt.savefig("files/data_distribution_data_train_separate_movies_aug_all.png")
    # plt.savefig("files/data_distribution_data_train_random_aug_rare.png")
    plt.savefig("files/data_distribution_data_train_separate_movies_aug_rare.png")

    return existing_single_genres


if __name__ == "__main__":
    # df = pd.read_csv("files/data_train_random_aug_all.tsv", sep="\t")
    # df = pd.read_csv("files/data_train_separate_movies_aug_all.tsv", sep="\t")
    # df = pd.read_csv("files/data_train_random_aug_rare.tsv", sep="\t")
    df = pd.read_csv("files/data_train_separate_movies_aug_rare.tsv", sep="\t")
    genres = get_genre_distribution(df)
    print(genres)
