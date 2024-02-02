import os

# import sys
# import pafy
# import cv2
# import youtube_dl
# import numpy as np


def create_dir(directory):
    # checking if the destination directory exists or not
    if not os.path.exists(directory):
        # if the destination directory is not present then create it
        os.makedirs(directory)


def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def choose_data_params(opt_frames, opt_augmentation, opt_data_distribution):
    if (
        opt_frames == "raw"
        and opt_augmentation == "none"
        and opt_data_distribution == "random"
    ):
        train_frames_type = "raw_frames"
        test_frames_type = "raw_frames"
        train_list = "data_train_random"
        test_list = "data_test_random"
    elif (
        opt_frames == "vec"
        and opt_augmentation == "none"
        and opt_data_distribution == "random"
    ):
        train_frames_type = "vec_frames"
        test_frames_type = "vec_frames"
        train_list = "data_train_random"
        test_list = "data_test_random"
    elif (
        opt_frames == "raw"
        and opt_augmentation == "all"
        and opt_data_distribution == "random"
    ):
        train_frames_type = "raw_frames_aug"
        test_frames_type = "raw_frames"
        train_list = "data_train_random_aug_all"
        test_list = "data_test_random"
    elif (
        opt_frames == "vec"
        and opt_augmentation == "all"
        and opt_data_distribution == "random"
    ):
        train_frames_type = "vec_frames_aug"
        test_frames_type = "vec_frames"
        train_list = "data_train_random_aug_all"
        test_list = "data_test_random"
    elif (
        opt_frames == "raw"
        and opt_augmentation == "rare"
        and opt_data_distribution == "random"
    ):
        train_frames_type = "raw_frames_aug"
        test_frames_type = "raw_frames"
        train_list = "data_train_random_aug_rare"
        test_list = "data_test_random"
    elif (
        opt_frames == "vec"
        and opt_augmentation == "rare"
        and opt_data_distribution == "random"
    ):
        train_frames_type = "vec_frames_aug"
        test_frames_type = "vec_frames"
        train_list = "data_train_random_aug_rare"
        test_list = "data_test_random"

    elif (
        opt_frames == "raw"
        and opt_augmentation == "none"
        and opt_data_distribution == "separate"
    ):
        train_frames_type = "raw_frames"
        test_frames_type = "raw_frames"
        train_list = "data_train_separate_movies"
        test_list = "data_test_separate_movies"
    elif (
        opt_frames == "vec"
        and opt_augmentation == "none"
        and opt_data_distribution == "separate"
    ):
        train_frames_type = "vec_frames"
        test_frames_type = "vec_frames"
        train_list = "data_train_separate_movies"
        test_list = "data_test_separate_movies"
    elif (
        opt_frames == "raw"
        and opt_augmentation == "all"
        and opt_data_distribution == "separate"
    ):
        train_frames_type = "raw_frames_aug"
        test_frames_type = "raw_frames"
        train_list = "data_train_separate_movies_aug_all"
        test_list = "data_test_separate_movies"
    elif (
        opt_frames == "vec"
        and opt_augmentation == "all"
        and opt_data_distribution == "separate"
    ):
        train_frames_type = "vec_frames_aug"
        test_frames_type = "vec_frames"
        train_list = "data_train_separate_movies_aug_all"
        test_list = "data_test_separate_movies"
    elif (
        opt_frames == "raw"
        and opt_augmentation == "rare"
        and opt_data_distribution == "separate"
    ):
        train_frames_type = "raw_frames_aug"
        test_frames_type = "raw_frames"
        train_list = "data_train_separate_movies_aug_rare"
        test_list = "data_test_separate_movies"
    elif (
        opt_frames == "vec"
        and opt_augmentation == "rare"
        and opt_data_distribution == "separate"
    ):
        train_frames_type = "vec_frames_aug"
        test_frames_type = "vec_frames"
        train_list = "data_train_separate_movies_aug_rare"
        test_list = "data_test_separate_movies"

    return train_frames_type, test_frames_type, train_list, test_list
