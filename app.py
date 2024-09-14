import streamlit as st
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import random

model_path = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_prompt(question):
    # Combine the conversation history into a formatted string
    history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])

    # Define prompt template
    base_template = """
    You are an empathetic and knowledgeable AI assistant specializing in mental health and dating advice. Focus on providing a single, detailed, and actionable piece of advice for the current question while considering the context provided below.

    Make sure your answer is:
    1. Directly related to the current question.
    2. Supportive, empathetic, and practical.
    3. Specific, concise, and non-repetitive.

    Avoid any unhelpful, generic, or repeated content.

    Conversation History:
    {conversation_history}

    Current Question:
    User: {question}

    Answer:
    """
    return base_template.format(conversation_history=history, question=question)

@st.cache_resource
def load_model():
    llm = GPT4All(model=model_path, callbacks=[], verbose=True)
    return llm

@st.cache_resource
def load_vectorstore():
    with open('data/mental_health.json', 'r', encoding='utf-8') as f:
        mental_health_data = json.load(f)
        mental_health_documents = [Document(page_content=entry['Context'] + "\n\n" + entry['Response']) for entry in mental_health_data]
    with open('data/dating_advice.txt', 'r', encoding='utf-8') as f:
        dating_advice_content = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    dating_advice_chunks = text_splitter.split_text(dating_advice_content)
    dating_advice_documents = [Document(page_content=chunk) for chunk in dating_advice_chunks]
    all_documents = mental_health_documents + dating_advice_documents

# Create embeddings
    embeddings = GPT4AllEmbeddings()
    vectorstore = Chroma(embedding_function=embeddings)
    batch_size = 5000
    for i in range(0, len(all_documents), batch_size):
        vectorstore.add_documents(all_documents[i:i + batch_size])

    return vectorstore

def clean_response(response):
    """
    Cleans the chatbot response by removing any unhelpful, redundant, or repetitive text,
    and stops at certain phrases that indicate the end of the response.
    """
    stop_phrases = ["Unhelpful Answer", "respond with a helpful answer", "If you don't know the answer", "Please provide", "Answer:"]
    lines = response.split('\n')
    seen_lines = set()
    filtered_lines = []

    for line in lines:
        # Stop if any stop phrase is encountered
        if any(stop_phrase in line for stop_phrase in stop_phrases):
            break
        if line.strip() and line not in seen_lines:  # Check for non-empty, unique lines
            filtered_lines.append(line.strip())
            seen_lines.add(line)
    cleaned_response = "\n".join(filtered_lines).strip()
    return cleaned_response

def interact_with_chatbot(question, qa):
    prompt = generate_prompt(question)
    raw_result = qa.invoke(input=prompt)
    formatted_result = raw_result.get("result", "No answer provided.")
    cleaned_result = clean_response(formatted_result)

    # Update conversation history
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": cleaned_result})

    return cleaned_result

def main():
    st.title("Chatbot for Dating and Mental Health Advice")
    llm = load_model()
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Load the question-answering chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Display chat messages from history on app 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user questions
    if query := st.chat_input("Enter your question:"):
        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Interact with the chatbot
        answer = interact_with_chatbot(query, qa)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)

if __name__ == "__main__":
    main()
