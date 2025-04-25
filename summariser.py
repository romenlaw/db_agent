import fitz
# from bs4 import BeautifulSoup
from langchain_text_splitters import HTMLSemanticPreservingSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
import os
import utils
from utils import client

OUTPUT_PATH = './input/summary'
system_prompt="""The documents are data dictionary data of DARE database tables.
               You are a SQL database expert. Use the documents to summarise the details of 
               the tables including: table name, business meaning of the table data, 
               column name, column data type, business meaning of column.
               When available, also summarise the source system or file information.
            """
class Summariser():
    def __init__(self, output_path=None):
        if output_path is None:
            self.output_path = OUTPUT_PATH
        else:
            self.output_path = output_path

        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
                print(f"Directory created: {self.output_path}")
            except OSError as e:
                print(f"Error creating directory {self.output_path}: {e}")
        return

    @staticmethod
    def summarise_pdf(pdf_path, chunk_size=8000, temperature=0.8):
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        # splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=100)
        chunks = splitter.split_text(text)
        table_name = str(pdf_path).split('-')[1]
        print(f"summarise_pdf got {len(chunks)} chunks, table name: {table_name}")

        queries = ["""summarise the content in details.
            """]
        summaries = []
        for i, chunk in enumerate(chunks):
            if i==0:
                queries=["""this is chunk #0 of the document, summarise the content in detail 
                         and memorise the table name and numbered list number in your context 
                         to be reused when processing subsequent chunks"""]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"}
                ]
            else:
                queries=[f"""this is chunk#{i} of the document which continues from previous chunks, 
                         share the context from chunk #0 for table name and listed number
                         and continue to summarise the content about the same table in detail.
                         Note that this chunk is a portion of the whole table information only."""]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"},
                    {"role": "user", "content": f"chunk #0 context: table name is {table_name}"}
                ]
            response = client.chat.completions.create(
                model=utils.CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                # max_tokens=1000
            ) 
        
            summaries.append(response.choices[0].message.content)
        return '\n\n'.join([str(s) for s in summaries])
    
    def summarise_pdf_directory(self, pdf_directory):
        directory = Path(pdf_directory)
        for file_path in directory.glob("*.pdf"):
            output_path = utils.get_basename_without_extension(file_path)
            output_path = f"{self.output_path}/{output_path}.txt"
            # print(file_path)
            summary = Summariser.summarise_pdf(file_path)
            with open(output_path, mode="w") as f:
                f.write(summary)

        return
        
    @staticmethod
    def summarise_html_content(html_content, chunk_size=8000, temperature=0.8):
        table_name = html_content.split('<title>')[1].split(' -')[0]
        splitter = HTMLSemanticPreservingSplitter(headers_to_split_on=['h1', 'h2', 'h3', 'h4'],
                                                  max_chunk_size=chunk_size, 
                                                  chunk_overlap=100,
                                                  separators=['div', 'tr', 'td'],
                                                  elements_to_preserve=['td'])
        chunks = splitter.split_text(html_content)
        
        print(f"summarise_html, got {len(chunks)} chunks, table name: {table_name}")
        queries = ["""summarise the content in detail
                   """]
        summaries = []
        
        for i, chunk in enumerate(chunks):
            if i==0:
                queries=["""this is chunk #0 of the document, summarise the content in detail 
                         and memorise the table name and numbered list number in your context 
                         to be reused when processing subsequent chunks"""]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"}
                ]
            else:
                queries=[f"""this is chunk#{i} of the document which continues from previous chunks, 
                         share the context from chunk #0 for table name and listed number
                         and continue to summarise the content about the same table in detail.
                         Note that this chunk is a portion of the whole table information only."""]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"},
                    {"role": "user", "content": f"chunk #0 context: table name is {table_name}"}
                ]
            response = client.chat.completions.create(
                model=utils.CHAT_MODEL,
                messages=messages,
                temperature=temperature
                # max_tokens=2048
            ) 
        
            summaries.append(response.choices[0].message.content)

        return '\n\n'.join([str(s) for s in summaries])
    
    @staticmethod
    def summarise_html(html_path, chunk_size=8000, temperature=0.8):
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return Summariser.summarise_html_content(html_content, chunk_size, temperature)
    
    def summarise_html_directory(self, html_directory):
        directory = Path(html_directory)
        for file_path in directory.glob("*.html"):
            output_path = utils.get_basename_without_extension(file_path)
            output_path = f"{self.output_path}/{output_path}.txt"
            # print(file_path)
            summary = Summariser.summarise_html(file_path)
            with open(output_path, mode="w") as f:
                f.write(summary)

        return
    
    @staticmethod
    def summarise_url(url):
        html_content = utils.get_confluence_page(url)
        return Summariser.summarise_html_content(html_content)
