import openai
import requests
import pandas as pd
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import numpy as np
import tiktoken


openai.api_key = "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"

# Read csv and parst into DF
df = pd.read_csv('Food_Scrap_Drop-Off_Locations_in_NYC.csv')
df['text'] = df.iloc[:, :7].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df = df[['text']].copy()  

# filter `data` based on the content of 'text' column
df = df[df['text'].str.len() > 0]  #remove any rows where 'text' is empty

# calculate cosin of embeddings and sort results

def get_rows_sorted_by_relevance(question, df):
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """

    # Get embeddings for the question text
    question_embeddings = get_embedding(question, engine=EMBEDDING_MODEL_NAME)

    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(
        question_embeddings,
        df_copy["embedding"].values,
        distance_metric="cosine"
    )

    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy

from dateutil.parser import parse

# Clean up text to remove empty lines and headings
df = df[(df["text"].str.len() > 0)]

# Create embeddings and add to dataframe

EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    # Send text data to OpenAI model to get embeddings
    response = openai.Embedding.create(
        input=df.iloc[i:i+batch_size]["text"].tolist(),
        engine=EMBEDDING_MODEL_NAME
    )

    # Add embeddings to list
    #print(response)
    embeddings.extend([data["embedding"] for data in response["data"]])

df["embedding"] = embeddings

def add_question_embeddings_and_sort(df, user_question):
    """
    Adds a 'distances' column to the DataFrame based on the cosine distance between
    the embeddings of 'user_question' and each row's embeddings in 'df'.
    Then, sorts the DataFrame by these distances.

    Parameters:
    - df: DataFrame containing a column 'embeddings' with pre-computed embeddings.
    - user_question: The user's question as a string.
    - embedding_model_name: The name of the OpenAI embedding model to use.

    Returns:
    - A DataFrame sorted by the distances to the user's question embeddings.
    """
    # Generate the embedding for the user question
    response = openai.Embedding.create(
        input=user_question,
        engine="text-embedding-ada-002"
    )
    question_embeddings = response["data"][0]["embedding"]

    # Calculate distances from the question embeddings to all row embeddings in the dataframe
    distances = distances_from_embeddings(
        question_embeddings,
        df["embedding"].tolist(),  # Ensure df["embeddings"] is in the correct format
        distance_metric="cosine"
    )

    # Add distances to the DataFrame and sort
    df["distances"] = distances
    #sorted_df = df.sort_values(by="distances", ascending=True)
    sorted_df = get_rows_sorted_by_relevance(user_question, df)

    return sorted_df

# Custom prompt creation

import tiktoken  # Ensure tiktoken is correctly installed and imported

def create_prompt(question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model.
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count the number of tokens in the prompt template and question
    prompt_template = """
    Answer the question based on the context below, and if the question
    can't be answered based on the context, say "I don't know."

    Context:

    {}

    ---

    Question: {}
    Answer:
    """

    current_token_count = len(tokenizer.encode(prompt_template.format("", question))) + \
                          len(tokenizer.encode(question))

    context = []
    for text in df.sort_values(by='distances', ascending=True)["text"].values:  # Assuming get_rows_sorted_by_relevance returns sorted DataFrame
        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count
        
        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), question)


# Answer question function

def answer_question(
    question, df, max_prompt_tokens=1800, max_answer_tokens=150
):
    COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"
    
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response, return the
    answer to the question according to an OpenAI Completion model. Format in a readable and user friendly form.
    If the model produces an error, return an empty string
    """

    prompt = create_prompt(question, df, max_prompt_tokens)
    #print(prompt)
    try:
        response = openai.Completion.create(
            model=COMPLETION_MODEL_NAME,
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# baseline model answer
    
user_prompt = """
Question: "Where can I drop off food scrapes in the Bronx on a Sunday?"
Answer:
"""
initial_user_answer = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=user_prompt,
    max_tokens=150
)["choices"][0]["text"].strip()
print("Baseline model answer: \n\n"+initial_user_answer+ "\n")

# list of test prompts
new_prompts = [
    "What are the food scrap drop-off options available in Brooklyn on a Sunday?",
    "Can you list any food scrap drop-off locations in Manhattan?",
    "Are there food scrap drop-off locations in the Bronx that are open 24/7?",
    "What food scrap drop-off services are available in Staten Island?",
    "I live in Queens, any food scrapes locations around me?"
]

# Function to iterate through the prompts and get answers
def get_answers_for_prompts(prompts, df):
    answers = []
    for prompt in prompts:
        df=add_question_embeddings_and_sort(df,prompt)
        answer = answer_question(prompt, df)
        answers.append((prompt, answer))
    return answers

# Call the function and print the answers
answers = get_answers_for_prompts(new_prompts, df)
for prompt, answer in answers:
    print(f"Prompt: {prompt}\nAnswer: {answer}\n\n")

