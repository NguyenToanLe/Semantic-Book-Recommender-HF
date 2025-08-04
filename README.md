# Semantic Book Recommender
This project is an interactive, content-based book recommendation system. It leverages semantic search to help users
discover new books by describing the type of book they are looking for, filtering by categories and emotional tones.

## Contents
1. [Features](#features)
2. [Technical Stack](#technical-stack)
3. [Project Structure](#project-structure)
4. [Local Installation & Usage](#local-installation-usage)
5. [Improvements](#improvements)
6. [Acknowledgement](#acknowledgement)

## 1. Features
- **Semantic Search:** Utilizes powerful LLMs to understand the meaning behind a user's query and find books with 
semantically similar descriptions.
- **Zero-shot Classification:** Utilizes [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) model to
simplify book categories, which were defined by the authors, into four categories, Fiction, Non-Fiction, Children Fiction,
and Children Non-Fiction.
- **Text Classification:** Utilizes [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
model to classify books into Ekman's 6 basic emotions (anger 🤬, disgust 🤢, fear 😨, joy 😀, sadness 😭, surprise 😲), 
and one additional neutral one.
- **Cosine Similarity:** This is the metric used to retrieve books from vector database.
- **Interactive Interface:** A user-friendly web interface built with Gradio allows for real-time recommendations.
- **Advanced Filtering:** Users can refine their search with multiple filters, including book categories and emotional 
tones (e.g., "Happy," "Suspenseful").
- **Deployment on Hugging Face Spaces:** The application is hosted as a publicly accessible web service, demonstrating a 
full deployment pipeline. A live application demo can be found [here](https://nguyentoanle41-book-recommender.hf.space/).

## 2. Technical Stacks
- **Python:** The core programming language for the entire project.
- **Gradio:** Used to create the user-friendly web interface for the recommendation engine.
- **LangChain:** Orchestrates the components of the recommendation pipeline, including text loading and semantic search.
- **ChromaDB:** A persistent vector database used to store and efficiently retrieve book embeddings. This is key for fast 
semantic search queries.
- **HuggingFace Embeddings:** The [FlagEmbedding](https://huggingface.co/BAAI/bge-base-en-v1.5) model is used to create 
high-quality vector embeddings of book descriptions.
- **Pandas:** For efficient data loading, cleaning, and manipulation of book metadata.

## 3. Project Structure
```
.
├── app.py                      # Main Gradio application file
├── requirements.txt            # Python dependencies for the project
├── Book-Cover-NaN.png          # Placeholder image for missing book covers
├── data_wrangling.ipynb        # This notebook is used to clean original books dataset and prepare data for embedding 
├── create_database.ipynb       # Notebook to create ChromaDB (This is no longer needed)
├── README.md                   # This README file
├── .gitignore                  # Specifies files/directories to be ignored by Git
└── dataset/
    ├── BooksCleaned.csv        # Cleaned book metadata
    ├── BooksDescription.txt    # Book descriptions for embedding
    └── books.csv               # Original books dataset
```

## 4. Local Installation & Usage
To run this project on your local machine, follow these steps:
### I. Clone the repository:
```Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```
### II. Create a virtual environment and activate it:
```Bash
# For Windows
python -m venv .venv
.venv\Scripts\activate

# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```
### III. Install the required dependencies:
```Bash
pip install -r requirements.txt
```
### IV. Run the application:
```Bash
python app.py
```
The Gradio application will now be running on a local URL (e.g., http://127.0.0.1:7860).

# 5. Improvements
- This project relies on [7k Books](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset. 
Hence, it just can recommend books in this dataset. If you want to embedd your own dataset, you have to transform your
dataset to the structure of this dataset, where `isbn13`, `title`, `description` are mandatory, and `subtitle`, 
`authors`, `categories` are optional.
- The application uses [FlagEmbedding](https://huggingface.co/BAAI/bge-base-en-v1.5) model as embedding engine. This is 
the reason why it just can understand and analyze English. Other languages are not supported and can lead to false or
hallucinated recommendations.
- Lastly, I tried to use abbreviations (such as WW1 for War World 1), but the embedding model does not understand these
phrases and led to hallucinated recommendations.

# 6. Acknowledgement
I would like to thank Dr. Jodie Burchell for her LLM course.