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

Loader = PyPDFLoader # Gán class loader tương ứng với định dạng tài liệu
FILE_PATH = r"D:\Project AIVN\Module 1\Project 1.2 Chatbox\Project 1.2_ Tạo và triển khai một chatbot cho một chủ đề cá nhân_[Description-Updated]-Project-RAG-Chatbot.pdf"
loader = Loader(FILE_PATH)
documents = loader.load() # Load tài liệu

embedding = HuggingFaceEmbeddings( model_name="bkai-foundation-models/vietnamese-bi-encoder") 
# Sử dụng mô hình embedding phù hợp với ngôn ngữ của tài liệu
semantic_chunker = SemanticChunker(embeddings=embedding,buffer_size=1,breakpoint_threshold_amount=95,
                                   breakpoint_threshold_type="percentile" ,
                                   min_chunk_size=500,add_start_index=True) 

# Cấu hình Semantic Chunking giúp chia nhỏ văn bản thành các phần có ý nghĩa, giúp tăng tính chính xác khi thực hiện truy vấn
docs = semantic_chunker.split_documents(documents)
print(f"Number of documents after semantic chunking: {len(docs)}")

# Tạo vector database từ các đoạn văn bản đã chia nhỏ
vector_db=Chroma.from_documents(documents=docs,
                                embedding=embedding)
retriever = vector_db.as_retriever()
while input("Nhấn Enter để tiếp tục truy vấn hoặc gõ 'exit' để thoát: ") != "exit":
    result = retriever.invoke(input = input("Nhập câu hỏi của bạn: "))
    print("Kết quả truy vấn:")
    print(result[0].page_content)
