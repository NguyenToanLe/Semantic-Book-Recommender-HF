from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import pandas as pd
import numpy as np
import gradio as gr


# def main():
# --------------------- Loading dataset and Initializations --------------------- #
books = pd.read_csv("./dataset/BooksCleaned.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "./Book-Cover-NaN.png",
    books["large_thumbnail"],
)
# Fill NaN in "authors" column
books["authors"] = books["authors"].fillna("")

raw_documents = TextLoader("dataset/BooksDescription.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
# db_books = Chroma(persist_directory="./dataset/chroma_db",
#                   embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

tone_mappings = {
    # From UI to Dataframe
    "Happy": "joy",
    "Surprising": "surprise",
    "Neutral": "neutral",
    "Angry": "anger",
    "Sad": "sadness",
    "Disgusting": "disgust",
    "Suspenseful": "fear"
}

# --------------------- Setup some logics for RAG --------------------- #
def retrieve_semantic_recommendations(
        query: str,
        categories: list = None,
        tones: list = None,
        k: int = 16
) -> pd.DataFrame:
    recommendations = db_books.similarity_search(query, k=k)
    isbn13_recommendations = [int(rec.page_content.strip('"').split()[0])  for rec in recommendations]
    books_recommendations = books[books["isbn13"].isin(isbn13_recommendations)].head(k)

    if categories != sorted(books["Predicted_Category"].unique()):
        books_recommendations = books_recommendations[books_recommendations["Predicted_Category"].isin(categories)]

    if tones != list(tone_mappings.keys()):
        books_recommendations.sort_values(by=[tone_mappings[tone] for tone in tones],
                                          ascending=False, inplace=True)

    return books_recommendations


def recommend_books(
        query: str,
        categories: list = None,
        tones: list = None,
):
    recommendations = retrieve_semantic_recommendations(query, categories, tones)
    rec_books = []

    for _, row in recommendations.iterrows():
        # We don't want to print the whole description.
        # If it has more than 30 characters, just print the first 30 characters and append with "..."
        description = row["description"].split(" ")
        description = " ".join(description[1:])         # Remove the isbn13 in description
        if len(description) > 30:
            description = description[:30] + "..."

        # 1 Author:     Author A
        # 2 Authors:    Author A and Author B
        # >=3 Authors:  Author A, Author B, Author C, and Author D
        authors = row["authors"].split(";")
        if len(authors) < 2:
            authors_str = row["authors"]
        elif len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        else:
            authors_str = ", ".join(authors[:-1])
            authors_str += f" and {authors[-1]}"

        # Create a Caption for current book
        if authors_str == "":
            caption = f"{row['title']}:\n {description}"
        else:
            caption = f"{row['title']} by {authors_str}:\n {description}"

        rec_books.append((row["large_thumbnail"], caption))

    return rec_books

# --------------------- GUI --------------------- #
categories = sorted(books["Predicted_Category"].unique())
tones = list(tone_mappings.keys())

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic Book Recommender")

        with gr.Row():
            user_query = gr.Textbox(label="Please describe the book you want to look for:",
                                    placeholder="e.g., A book about wizard")
            category_dropdown = gr.Dropdown(choices=categories,
                                            label="Select one more more category:",
                                            value=["Fiction"],
                                            multiselect=True)
            tone_dropdown = gr.Dropdown(choices=tones,
                                        label="Select one more more category:",
                                        value=["Happy"],
                                        multiselect=True)
            submit_button = gr.Button("Find recommendations")

        gr.Markdown("## Recommendations")
        output = gr.Gallery(label="Recommended Books", columns=8, rows=2)

        submit_button.click(fn=recommend_books,
                            inputs=[user_query, category_dropdown, tone_dropdown],
                            outputs=output)


if __name__ == "__main__":
    # main()
    dashboard.launch(share=True)
    # print("Finished")
