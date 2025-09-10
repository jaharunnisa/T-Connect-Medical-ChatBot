# %%
print("OK")

# %%
%pwd

# %%
import os 
os.chdir("../")

# %%
%pwd

# %%
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# %%
# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# %%
extracted_data = load_pdf_files("data")

# %%
extracted_data

# %%
len(extracted_data)

# %%
from typing import List
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# %%
minimal_docs = filter_to_minimal_docs(extracted_data)

# %%
minimal_docs

# %%
def text_split(minimal_docs):
    # Use separators to preserve semantic structure (paragraphs, headings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", " "]
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# %%
texts_chunk = text_split(minimal_docs)
print(f"Number of chunks: {len(texts_chunk)}")

# %%
texts_chunk

# %%
from langchain.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

embedding = download_embeddings()

# %%
embedding

# %%
vector = embedding.embed_query("Hello world")
vector

# %%
print( "Vector length:", len(vector))

# %%
from dotenv import load_dotenv
import os
load_dotenv()

# %%
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")




# %%
from pinecone import Pinecone 
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east1-gcp")  # adjust region



# %%
pc

# %%
from pinecone import Pinecone, ServerlessSpec

# Create Pinecone client
pc = Pinecone(
    api_key=API_KEY,
    environment="us-east1-gcp"  # Make sure this matches your Pinecone project
)

# Name of your index
index_name = "medical-chatbot"

# List existing indexes
print("Existing indexes:", pc.list_indexes().names())

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,            # must match your embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created successfully!")

# Connect to the index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")




# %%
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)



from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env
load_dotenv()

# Set API key securely
pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# Connect to existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)



# %%
dswith = Document(
    page_content="dswithbappy is a youtube channel that provides tutorials on various topics.",
    metadata={"source": "Youtube"}
)

# %%
docsearch.add_documents(documents=[dswith])

# %%
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# %%
retrieved_docs = retriever.invoke("What is Acne?")
retrieved_docs

# %%
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# %%
# Step 1: Import HuggingFace tools
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# Step 2: Load your SLM model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # replace with your SLM model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512  # you can adjust
)

# Step 4: Wrap it in LangChain's HuggingFacePipeline
chatModel = HuggingFacePipeline(pipeline=pipe)

# %%
system_prompt = (
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# %%
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# %%
response = rag_chain.invoke({"input": "what is Acne?"})
print(response["answer"])

# %%
response = rag_chain.invoke({"input": "what is the Treatment of Acne?"})
print(response["answer"])

# %%



