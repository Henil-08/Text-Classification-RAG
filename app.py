import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
import time
import logging

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    filename="app.log", 
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s"
)

# Function to measure response time
def measure_response_time(start_time):
    end_time = time.time()
    return end_time - start_time

file_path = 'Data/knowledge_base.txt'

## Set up Streamlit 
st.set_page_config(page_title="Customer Support")
st.title("Support Ticket Classifier")

## Environment variables
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Check environment variables
if not (os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
    st.error("API tokens not set. Please check your environment variables.")
    logging.error("API tokens missing.")
else:
    try:
        ## Load the Model and embeddings
        llm = ChatGroq(model_name="llama-3.1-8b-instant")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        ## Enter support ticket
        support = st.text_input("Enter the Support Ticket:")

        if support:
            start_time = time.time()

            try:
                # Load the knowledge_base
                loader = TextLoader(file_path)
                docs = loader.load()
                
                # Split and create embeddings for the documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
                splits = text_splitter.split_documents(docs)
                vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                retriever = vectorstore.as_retriever(k=3) 

                # Prompt
                system_prompt = (
                    """
                    ### Instruction
                    Context:
                    {context}

                    Understand the context containing labels and their description and then.

                    Classify the input text below into one of the following labels based on the context:
                    Category 1 - Login support
                    Category 2 - App Functionality
                    Category 3 - Billing 
                    Category 4 - Account Management
                    Category 5 - Performance support
                    

                    If you dont know the answer or the input is irrelevant to the context just say I don't know. Just output the label and nothing else. 
                    """
                )

                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )

                # Create RAG chain
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke({"input": support})
                response_time = measure_response_time(start_time)

                st.success(f"Response time: {response_time:.2f} seconds")
                st.write()
                st.write(response['answer'])
                logging.info(f"Processed support successfully in {response_time:.2f} seconds.")

            except Exception as e:
                st.error("An error occurred while processing the request.")
                logging.error(f"Error during processing: {str(e)}")

        else:
            st.info("Please enter the support ticket.")
            logging.debug("No support ticket entered by the user.")

    except Exception as e:
        st.error("An error occurred while setting up the application.")
        logging.error(f"Application setup error: {str(e)}")



