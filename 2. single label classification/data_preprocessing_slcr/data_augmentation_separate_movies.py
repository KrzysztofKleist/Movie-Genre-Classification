import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import random

import os
import shutil

from utilities import create_dir

import csv

from tqdm import tqdm


def generate_random_parameters(im_width, im_height):
    width = random.randint(im_width / 2, im_width)
    height = random.randint(im_height / 2, im_height)

    brightness = random.uniform(0.2, 0.8)
    contrast = random.uniform(0.2, 0.8)
    saturation = random.uniform(0.2, 0.8)
    hue = random.uniform(0.1, 0.2)

    blur_kernel_size = random.choice(range(3, 100, 2))

    return (height, width, brightness, contrast, saturation, hue, blur_kernel_size)


def copy_image_to_new_dir(image_path):
    # Extract directory path
    image_path = image_path.replace("\\", "/")
    directory_path = os.path.dirname(image_path)

    # Create augmented directory path
    augmented_directory_path = directory_path.replace("_frames", "_frames_aug")
    create_dir(augmented_directory_path)

    # Create augmented image file path
    augmented_image_path = image_path.replace("_frames", "_frames_aug")

    
    shutil.copyfile(image_path, augmented_image_path)


def generate_modified_image(image_path):
    # Load the image
    image_path = image_path.replace("\\", "/")
    image = Image.open(image_path)
    width, height = image.size

    (
        crop_height,
        crop_width,
        brightness,
        contrast,
        saturation,
        hue,
        blur_kernel_size,
    ) = generate_random_parameters(width, height)

    # Define transformations to apply to the image
    transform = transforms.Compose(
        [
            transforms.RandomCrop((crop_height, crop_width)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ),
            transforms.GaussianBlur(kernel_size=blur_kernel_size),
        ]
    )

    # Transform the image
    transformed_image = transform(image)

    # # Display the transformed image
    # plt.imshow(transformed_image)
    # plt.axis("off")
    # plt.show()

    # Extract directory path
    directory_path = os.path.dirname(image_path)

    # Create augmented directory path
    augmented_directory_path = directory_path.replace("_frames", "_frames_aug")
    create_dir(augmented_directory_path.replace("\\", "/"))

    # Create augmented image file path
    augmented_image_path = image_path.replace("_frames", "_frames_aug").replace(
        ".jpg", "_aug.jpg"
    )

    transformed_image.save(augmented_image_path.replace("\\", "/"))

    return augmented_image_path.replace("raw_frames_aug/", "").replace("\\", "/")


# Generates augmented images with focus on rarest genres
# Horror, Crime, Thriller, Romance
def main():
    # Initialize an empty list to store rows
    data = []

    # Open the TSV file
    # Choose one of these two sets of initializers accordingly
    # # data_train_random.tsv
    # reader_tsv_file_path = "files_slcr/data_train_random.tsv"
    # writer_tsv_file_path = "files_slcr/data_train_random_aug_all.tsv"
    # data_test_separate_movies.tsv
    reader_tsv_file_path = "files_slcr/data_train_separate_movies.tsv"
    writer_tsv_file_path = "files_slcr/data_train_separate_movies_aug_all.tsv"

    with open(reader_tsv_file_path, "r", newline="", encoding="utf-8") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")

        counter = 0

        # Read and store each row
        for row in tsv_reader:
            data.append(row)

    # Open the data_train_random_aug_all.tsv file for writing
    # It generates new images depending on the genre
    with open(writer_tsv_file_path, "w", newline="", encoding="utf-8") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter="\t")
        counter = 0

        for row in tqdm(data):
            if counter == 0:
                tsv_writer.writerow(row)
            elif row[2] == "Drama":
                copy_image_to_new_dir("raw_frames/" + row[1])
                tsv_writer.writerow(row)
                # Check if the random number is less than 0.9 then do the augmentation
                if random.random() < 0.9:
                    tsv_writer.writerow(
                        [
                            row[0],
                            generate_modified_image("raw_frames/" + row[1]),
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            row[6],
                        ]
                    )
            elif row[2] == "Com_Rom":
                copy_image_to_new_dir("raw_frames/" + row[1])
                tsv_writer.writerow(row)
                # Check if the random number is less than 0.9 then do the augmentation
                if random.random() < 0.3:
                    tsv_writer.writerow(
                        [
                            row[0],
                            generate_modified_image("raw_frames/" + row[1]),
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            row[6],
                        ]
                    )
            elif row[2] == "Thr_Hor_Cri":
                copy_image_to_new_dir("raw_frames/" + row[1])
                tsv_writer.writerow(row)
                # Check if the random number is less than 0.9 then do the augmentation
                if random.random() < 0.4:
                    tsv_writer.writerow(
                        [
                            row[0],
                            generate_modified_image("raw_frames/" + row[1]),
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            row[6],
                        ]
                    )
            elif row[2] == "Act_Adv":
                copy_image_to_new_dir("raw_frames/" + row[1])
                tsv_writer.writerow(row)
                # Check if the random number is less than 0.9 then do the augmentation
                if random.random() < 0.1:
                    tsv_writer.writerow(
                        [
                            row[0],
                            generate_modified_image("raw_frames/" + row[1]),
                            row[2],
                            row[3],
                            row[4],
                            row[5],
                            row[6],
                        ]
                    )
            counter += 1


if __name__ == "__main__":
    main()
