import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# Bỏ quantization vì bitsandbytes không hỗ trợ GPU/CPU hiện tại
# Sử dụng float16 để tiết kiệm memory

MODEL_NAME = "lmsys/vicuna-7b-v1.5"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto"
)

llm = HuggingFacePipeline(
    pipeline=model_pipeline,
)

prompt = hub.pull("rlm/rag-prompt")

# ============= LOAD AND PROCESS DOCUMENTS =============
FILE_PATH = r"D:\Project AIVN\Module 1\Project 1.2 Chatbox\Project 1.2_ Tạo và triển khai một chatbot cho một chủ đề cá nhân_[Description-Updated]-Project-RAG-Chatbot.pdf"

loader = PyPDFLoader(FILE_PATH)
documents = loader.load()

# ============= CREATE EMBEDDINGS =============
embedding = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

# ============= SPLIT DOCUMENTS SEMANTICALLY =============
semantic_chunker = SemanticChunker(
    embeddings=embedding,
    buffer_size=1,
    breakpoint_threshold_amount=95,
    breakpoint_threshold_type="percentile",
    min_chunk_size=500,
    add_start_index=True
)

docs = semantic_chunker.split_documents(documents)
print(f"Number of documents after semantic chunking: {len(docs)}")

# ============= CREATE VECTOR DATABASE =============
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding
)

retriever = vector_db.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

USER_QUESTION = "YOLOv10 là gì?"
output = rag_chain.invoke(USER_QUESTION)
print(f"Output: {output}")

# Xử lý output an toàn
if "Answer:" in output:
    answer = output.split("Answer:")[1].strip()
else:
    answer = output

print(f"Vicuna: {answer}")