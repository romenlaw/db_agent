import os
import fitz
import faiss
import numpy as np
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import CharacterTextSplitter
from utils import client

DEFAULT_CHUNK_SIZE = 2000
MEMORY_PATH = './memory'

class Embedder():
    def __init__(self, model='text-embedding-3-large_v1',
                 chunk_size=DEFAULT_CHUNK_SIZE,
                 mem_path=MEMORY_PATH):
        self.model = model
        self.chunk_size = chunk_size
        self.mem_path = mem_path
        self._reset_memory()
        if not os.path.exists(self.mem_path):
            try:
                os.makedirs(self.mem_path)
                print(f"Directory created: {self.mem_path}")
            except OSError as e:
                print(f"Error creating directory {self.mem_path}: {e}")

    def _reset_memory(self):
        self.chunks = []
        self.embeddings = np.array([])
        self.indices = None

    def _save_memory(self):
        """Save chunks and embeddings into files on disk"""
        np.save(f'{self.mem_path}/embeddings', self.embeddings)
        with open(f'{self.mem_path}/chunks.txt', 'w', encoding='utf-8') as f:
            f.write('\n===\n'.join(self.chunks))

        faiss.write_index(self.indices, f'{self.mem_path}/faiss_index.bin')

    def load_memory(self):
        self.embeddings = np.load(f'{self.mem_path}/embeddings.npy')
        with open(f'{self.mem_path}/chunks.txt', 'r', encoding='utf-8') as f:
            self.chunks = f.read().split('\n===\n')
        self.indices = faiss.read_index(f'{self.mem_path}/faiss_index.bin')
        

    def embed_chunks(self, chunks):
        """Embed the chunks using the class' embedding model"""
        # if the chunks is too large the model will throw exception, so 
        # break it into sub-chunks of 96 elements each
        step = 96
        embeddings = None
        for i in range(0, len(chunks), step):
            sub_chunks = chunks[i:i+step]
            response = client.embeddings.create(input=sub_chunks, model=self.model)
            sub_embeddings = np.array([item.embedding for item in response.data])
            embeddings = sub_embeddings if embeddings is None else np.vstack((embeddings, sub_embeddings))

        return embeddings
    
    def embed_pdf(self, pdf_file_path, chunk_size=None):
        """Read and convert PDF file into text, then chunk it, embed it.
        Args:
            pdf_file_path (str): file path and file name of the PDF file.
            chunk_size (int): chunking size.
        Returns:
            chunks: list of texts after chunking
            embeddings: list of embedded vectors
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        chunks = []

        doc = fitz.open(pdf_file_path)
        text = "".join(page.get_text() for page in doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        # splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = self.embed_chunks(chunks=chunks)

        return chunks, embeddings
    
    def _embed_directory(self, file_directory, file_type='txt'):
        directory = Path(file_directory)
        for file_path in directory.glob(f"*.{file_type}"):
            if file_type=='pdf':
                chunks, embeddings = self.embed_pdf(file_path)
            elif file_type=='html':
                chunks, embeddings = self.embed_html(file_path)
            else:
                chunks, embeddings = self.embed_txt(file_path)
            self.chunks += chunks
            
            if self.embeddings.size > 0:
                self.embeddings = np.vstack((self.embeddings, embeddings)) 
            else: 
                self.embeddings = embeddings
        return

    def embed_directory(self, file_directory, file_type='txt'):
        """Read all files in the given path, for each file, chunk and embed it.
        At the end of it, concat all the chunks and embeddings into self.chunks and self.embeddings.
        Args:
            file_directory (str): directory where the files are stored.
            file_type (str): 'txt' | 'pdf' | 'html'
        Returns:
            chunks: list of texts after chunking
            embeddings: list of embedded vectors
        """
        assert file_type=='txt' or file_type=='pdf' or file_type=='html', \
            "Invalid file type, must be txt, html or pdf"
        
        # embed the common files first
        self._embed_directory('./input/summary')
        self._embed_directory(file_directory, file_type)
        
        self.indices = self.create_faiss_index(self.embeddings)
        self._save_memory()
        return

    def embed_txt(self, txt_file_path, chunk_size=None):
        """Read text file, then chunk it, embed it.
        Args:
            txt_file_path (str): file path and file name of the text file.
            chunk_size (int): chunking size.
        Returns:
            chunks: list of texts after chunking
            embeddings: list of embedded vectors
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        chunks = []
        
        with open(txt_file_path, 'r') as f:
            text = f.read()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = self.embed_chunks(chunks=chunks)

        return chunks, embeddings

    def embed_html(self, html_file_path, chunk_size=None):
        """Read text file, then chunk it, embed it.
        Args:
            html_file_path (str): file path and file name of the html file.
            chunk_size (int): chunking size.
        Returns:
            chunks: list of texts after chunking
            embeddings: list of embedded vectors
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        chunks = []
        
        with open(html_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        chunks = splitter.split_text(text)
        embeddings = self.embed_chunks(chunks=chunks)

        return chunks, embeddings

    
    @staticmethod
    def create_faiss_index(embeddings):
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)  # L2 distance for similarity
        index.add(np.array(embeddings))  # Add embeddings to index
        return index
    
    def search_chunks(self, queries):
        """queries - list of questions/strings
        chunks - list of chunks
        """
        # query_embedding = np.array(get_embedding(query)).reshape(1, -1)
        query_embedding = self.embed_chunks(queries)
        distances, indices = self.indices.search(query_embedding, k=10)  # Retrieve top k relevant chunks
        return [self.chunks[i] for i in indices[0]]
