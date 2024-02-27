import openai
import requests
import pandas as pd
from openai.embeddings_utils import get_embedding, distances_from_embeddings
import numpy as np
import tiktoken

# add your OpenAI apikey
openai.api_key = "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"

# read csv and parse into DF. Merge first seven colunns into a single column
df = pd.read_csv('Food_Scrap_Drop-Off_Locations_in_NYC.csv')
df['text'] = df.iloc[:, :7].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df = df[['text']].copy()  

# filter `data` based on the content of 'text' column
df = df[df['text'].str.len() > 0]  #remove any rows where 'text' is empty

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

    embeddings.extend([data["embedding"] for data in response["data"]])

df["embedding"] = embeddings

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
    for text in df.sort_values(by='distances', ascending=True)["text"].values:
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

# function for baseline answer

def get_baseline_answer(question):
    """
    Generate a baseline answer for a given question using the GPT-3.5-turbo-instruct model.
    """
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"Question: {question}\nAnswer:",
            max_tokens=150
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return "An error occurred while generating the baseline answer."

# Function for Q&A pairs baseline and improved

def generate_qa_pairs(prompts, df):
    """
    For each prompt, generate Q&A pairs for both baseline and improved answers.
    """
    qa_pairs = []
    for prompt in prompts:
        # Get baseline answer
        baseline_answer = get_baseline_answer(prompt)
        
        # Get improved answer
        df = add_question_embeddings_and_sort(df, prompt)
        improved_answer = answer_question(prompt, df)
        
        # Append both Q&A pairs
        qa_pairs.append({
            "question": prompt,
            "baseline_answer": baseline_answer,
            "improved_answer": improved_answer
        })
    return qa_pairs

# Example usage
my_prompts = [
    "Where can I drop off food scrapes in the Bronx on a Friday?",
    "Cam I drop of my food scraps in Brooklyn on a Sunday, if Yes when and where?"
    # Add more prompts as needed
]

# Call the function and print the Q&A pairs
qa_pairs = generate_qa_pairs(my_prompts, df)
for pair in qa_pairs:
    print(f"\nQuestion: {pair['question']}")
    print(f"\nBaseline Answer:\n {pair['baseline_answer']}")
    print(f"\nImproved Answer:\n {pair['improved_answer']}\n")
