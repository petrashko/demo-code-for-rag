import os
#
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
#from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

#working_dir = os.path.dirname(os.path.abspath(__file__))
vectordb_directory = 'vectordb_dir'


def get_embeddings():
    #model_id = 'sentence-transformers/all-MiniLM-l6-v2'
    model_id = 'BAAI/bge-m3'

    #model_args = {'device': 'cuda'}
    model_args = {'device': 'cpu'}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_id,
        model_kwargs=model_args,
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


huggingface_embeddings = get_embeddings()

#loader = DirectoryLoader(path='data_pdf', glob='./*.pdf', loader_cls=UnstructuredFileLoader)
loader = PyPDFDirectoryLoader(path='data_pdf')
documents = loader.load()

#text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

text_chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=huggingface_embeddings,
    persist_directory=vectordb_directory
)


if __name__ == '__main__':
    print("Documents Vectorized"); print()
    print('END!!!')

