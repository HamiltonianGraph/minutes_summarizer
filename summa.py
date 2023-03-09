import os
import openai
import pypdf
from langchain.llms import OpenAIChat
from langchain.vectorstores import FAISS
from langchain.chains import VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader

def split_pdf(fpath,chunk_chars=3000,overlap=50):
    """ Pre-process the PDF into chunks of 3000 tokens
    """
    pdfReader = pypdf.PdfReader(fpath)
    splits = []
    split = ""
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        while len(split) > chunk_chars:
            splits.append(split[:chunk_chars])
            split = split[chunk_chars - overlap :]
    if len(split) > overlap:
        splits.append(split[:chunk_chars])
    return splits

def create_vector_db(splits):
    """ Create a vecctor DB index of the PDF """
    embeddings = OpenAIEmbeddings(openai_api_key="<YOUR-API-KEY>")
    return FAISS.from_texts(splits,embeddings)

splits = split_pdf('testminutes.pdf')
embed = create_vector_db(splits)
llm = OpenAIChat(temperature=0,openai_api_key="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")
chain = VectorDBQA.from_chain_type(llm, chain_type="stuff", vectorstore=embed)
query = "List and summarize the decision items in the meeting and the decision the board came to on them"
response = chain.run(query)
print(response)
