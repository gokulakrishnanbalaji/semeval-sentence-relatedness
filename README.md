
# Semantic Textual Similarity App

This application measures the semantic textual similarity between two sentences. It uses a pre-trained Sentence Transformer model to generate embeddings for input sentences and calculates the similarity score.

## Installation

1. Install the required dependencies:

```
pip install -r requirements.txt
```

2. Make sure to create folder/directories called `models`, `predictions` and `zipped_predictions` for the output to get savedin the local disk.
```
mkdir models
```
```
mkdir predictions
```
```
mkdir zipped_predictions
```

## Usage

1. Run the application using the command:

```
python3 app.py
```

2. To use the graphical user interface (GUI) for testing the model, run:

```
python gui.py
```

The GUI allows you to interactively test the model by providing two text inputs and observing the similarity score.

Feel free to explore and experiment with different sentences in your chosen language.

**Note:** Ensure that you have an active internet connection for the first run, as the Sentence Transformer model may need to be downloaded.


Make sure to replace `{lang}` with the actual file name corresponding to the language you are working with (e.g., `tel.py`, `tam.py`, etc.). 