# Movie Genre Classification - posters only

## Initial steps:

1. Creating, cleaning and sorting the dataset in separate parts - [create_dataset.ipynb](https://github.com/KrzysztofKleist/Movie-Genre-Classification/blob/main/create_dataset.ipynb) \*
2. Running the data through the vectorscope function - [vectorscope_data.ipynb](https://github.com/KrzysztofKleist/Movie-Genre-Classification/blob/main/vectorscope_data.ipynb) \*
3. Running the AlexNet model with the original, cleaned data - [alexnet_original_data.ipynb](https://github.com/KrzysztofKleist/Movie-Genre-Classification/blob/main/alexnet_original_data.ipynb) \*\*
4. Running the AlexNet model with the data modified by the vectorscope - [alexnet_vectorscope_data.ipynb](https://github.com/KrzysztofKleist/Movie-Genre-Classification/blob/main/alexnet_vectorscope_data.ipynb) \*\*

\* At this point the data has to be packed to the .zip file, also with train.txt as the list of labels. Then the .zip files need to be uploaded to google.drive as google.colab was used to run the first tests with AlexNet model.

\*\* Done in google.colab.

### Questions:

- Should the problem be considered multi-label classification (more than one prediction allowed in the output)?
