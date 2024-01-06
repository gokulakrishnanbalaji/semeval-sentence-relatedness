
# Semantic Textual Similarity App

This application measures the semantic textual similarity between two sentences. It uses a pre-trained Sentence Transformer model to generate embeddings for input sentences and calculates the similarity score.

## Installation

1. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Run the application using the command:

```
python3 {lang}.py
```

Replace `{lang}` with any one of the languages specified in the problem statement (e.g., `tel.py`, `tam.py`, `mal.py`, `kan.py`, `hin.py`, `ben.py`, `eng.py`).

2. To use the graphical user interface (GUI) for testing the model, run:

```
python gui.py
```

The GUI allows you to interactively test the model by providing two text inputs and observing the similarity score.

Feel free to explore and experiment with different sentences in your chosen language.

**Note:** Ensure that you have an active internet connection for the first run, as the Sentence Transformer model may need to be downloaded.


Make sure to replace `{lang}` with the actual file name corresponding to the language you are working with (e.g., `tel.py`, `tam.py`, etc.). 