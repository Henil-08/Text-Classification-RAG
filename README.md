# **Customer Support Ticket Classification with RAG**

## **Overview**

This project implements a text classification system using a Retrieval-Augmented Generation (RAG) approach. The goal is to categorize customer support tickets into predefined categories by leveraging a pre-trained large language model and a custom-built knowledge base.

## **Project Structure**

- `app.py`: Main Streamlit application implementing the RAG approach for customer support ticket classification.
- `research.ipynb`: Jupyter notebook detailing the research and exploration conducted during development.
- `requirements.txt`: File listing all dependencies required to run the application.
- `Data/knowledge_base.txt`: Text file containing the knowledge base used for classification.

## **Usage**
- Input a customer support ticket into the app, and it will classify the ticket into one of the following categories:
    1. Login Issues
    2. App Functionality
    3. Billing
    4. Account Management
    5. Performance Issues
- The system uses a RAG approach to retrieve relevant documents from the knowledge base and assist the language model in generating an accurate classification.

## **Approach**
### 1. **Data Preparation**
The knowledge base was created to include detailed descriptions for five categories relevant to customer support. The given example knowledge base was stored as a text file for better loading. Used `TextLoader` class of langchain to load the given knowledge base as a document. Used `RecursiveCharacterTextSplitter` to split the document into chunks of fixed size.  

### 2. **Retrieval**
To enhance the classification accuracy, I employed FAISS *(Facebook AI Similarity Search)* to index the knowledge base and the example dataset. When a new support ticket is input, the system retrieves the 3 most relevant examples and category definitions based on semantic similarity, providing the model with contextual guidance.

### 3. **Prompt Design**
I structured the prompt to include the following:
- A list of relevant category labels and their definitions retrieved from the knowledge base.
- A list of the total classification labels.
- The input text requiring classification.

This design allows the model to understand from context and labels, leading to more accurate and context-aware classifications.

### 4. **Model Inference**
The final step involves creating a RAG chain by querying a pre-trained large language model (Llama 3.1-8b-instant) with the structured prompt. I have used Groqcloud LPU engine as it is faster in providing the response and is free. The model generates the most appropriate category label for the input ticket based on the provided context.

## **Rationale**
The rationale behind this approach lies in enhancing the model's understanding through retrieval. By providing the model with labels and definitions closely related to the input, I aimed to improve its contextual comprehension and reduce the likelihood of misclassification.

## **Results**
The system demonstrated effective classification of support tickets into the correct categories. While the overall performance was robust, it was particularly strong in scenarios where the retrieved examples closely matched the input text.
Here’s an example of how it performed on different tickets:

- Input Ticket: "The app crashes very often."
    Model Output: Category 2 - App Functionality
- Input Ticket: "Refund not processed."
    Model Output: Category 3 - Billing

These examples illustrate the system's capability to leverage relevant context to accurately categorize different types of support tickets.

## **Potential Shortcomings and Improvements**
1. **Knowledge Base Expansion:** The current knowledge base is static and loaded from a single file. In a production scenario, we might need a more dynamic approach where the knowledge base can be updated or expanded without restarting the application. Expanding the knowledge base with more diverse examples could further improve classification accuracy, especially for edge cases.
2. **Fine-Tuning the Model:** Fine-tuning the language model using QLORA on a domain-specific dataset could enhance its understanding of nuanced language patterns in customer support tickets.
3. **Advanced Retrieval Techniques:** Incorporating more sophisticated retrieval techniques (Hybrid Search/Reranker) or a larger vector store (AstraDB/Pinecone) could lead to better document retrieval, thereby providing even more relevant context for classification.
4. **Continuous Learning:** Integrating a feedback loop where misclassifications are reviewed and corrected could help iteratively improve the model's performance over time.

## **Conclusion**
This project successfully demonstrates the use of a RAG approach for text classification, leveraging retrieval to enhance the performance of a large language model. While the current solution is effective, there remains significant potential for refinement and optimization.