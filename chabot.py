from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain.schema import Document

model_path = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# Create callback handler
class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end='', flush=True)

llm = GPT4All(model=model_path, callbacks=[MyCallbackHandler()], verbose=True)

# Initialize conversation history
messages = []

# Define prompt template
template = """
You are an empathetic and knowledgeable AI assistant specializing in mental health and dating advice. 
Focus on providing a single, detailed, and actionable piece of advice for the current question while considering the context provided below.

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

# Create the prompt variable
def generate_prompt(question):
    history = "\n".join( messages[-5:])  
    return template.format(messages=history, question=question)

# Load data
with open('data/mental_health.json', 'r', encoding='utf-8') as f:
    mental_health_data = json.load(f)
mental_health_documents = [Document(page_content=entry['Context'] + "\n\n" + entry['Response']) for entry in mental_health_data]

with open('data/dating_advice.txt', 'r', encoding='utf-8') as f:
    dating_advice_content = f.read()

# Split the content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
dating_advice_chunks = text_splitter.split_text(dating_advice_content)
dating_advice_documents = [Document(page_content=chunk) for chunk in dating_advice_chunks]

all_documents = mental_health_documents + dating_advice_documents

# Create a vector store and add the text chunks
embeddings = GPT4AllEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

# Add documents to the vector store in batches
batch_size = 5000
for i in range(0, len(all_documents), batch_size):
    vectorstore.add_documents(all_documents[i:i + batch_size])

# Create a retriever with fewer retrieved documents
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Load the question-answering chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# Clean response
def clean_response(response):
    """
    Cleans the chatbot response by removing any unhelpful or redundant text and filtering out repeated content.
    """
    lines = response.split('\n')
    seen_lines = set()
    filtered_lines = []
    for line in lines:
        if ("Unhelpful Answer" not in line and
            "respond with a helpful answer" not in line and
            "If you don't know the answer" not in line and
            line not in seen_lines):
            filtered_lines.append(line)
            seen_lines.add(line)

    cleaned_response = "\n".join(filtered_lines).strip()
    return cleaned_response

def truncate_response(response, max_length=500):
    """
    Truncates the response to a maximum number of characters to prevent redundancy.
    """
    if len(response) > max_length:
        truncated = response[:max_length]
        last_period_index = truncated.rfind('.')
        if last_period_index != -1:
            return truncated[:last_period_index + 1].strip()
        last_space_index = truncated.rfind(' ')
        if last_space_index != -1:
            return truncated[:last_space_index].strip()
        return truncated.strip()
    return response

# Function to interact with the chatbot
def interact_with_chatbot(question):
    global messages
    prompt = generate_prompt(question)
    raw_result = qa.invoke(input=prompt)
    
# Post-process the result
    formatted_result = raw_result.get("result", "No answer provided.")
    cleaned_result = clean_response(formatted_result)
    final_result = truncate_response(cleaned_result)

# Update conversation history
    messages.append(f"User: {question}")
    messages.append(f"Assistant: {final_result}")

    return final_result

# First question
query1 = "I'm feeling very down lately, what should I do?"
answer1 = interact_with_chatbot(query1)
print(f"Answer: {answer1}\n")

# Follow-up question 1
query2 = "What should I do if I feel nervous about my first date tomorrow?"
answer2 = interact_with_chatbot(query2)
print(f"Answer: {answer2}\n")

# Follow-up question 2
query3 = "Can you give me some advice on what to wear on a first date?"
answer3 = interact_with_chatbot(query3)
print(f"Answer: {answer3}\n")
