import fitz
# from bs4 import BeautifulSoup
from langchain_text_splitters import HTMLSemanticPreservingSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
import os
import utils
from utils import client

OUTPUT_PATH = './input/summary'
SYSTEM_PROMPT="""The documents are data dictionary data of DARE database tables.
               You are a SQL database expert. Use the documents to summarise the details of 
               the tables including: table name, business meaning of the table data, 
               column name, column data type, business meaning of column.
               When available, also summarise the source system or file information.
            """
CHUNK0_QUERY="""this is chunk #0 of the document, summarise the content in detail 
                and memorise the table name and numbered list number in your context 
                to be reused when processing subsequent chunks. Make sure all your output is in utf-8 encoding."""
CHUNK_QUERY="""of the document which continues from previous chunks, 
            share the context from chunk #0 for table name and listed number
            and continue to summarise the content about the same table in detail.
            Note that this chunk is a portion of the whole table information only.
            Make sure all your output is in utf-8 encoding."""
class Summariser():
    def __init__(self, output_path=None, system_prompt=None, chunk0_query=None, chunk_query=None):
        if output_path is None:
            self.output_path = OUTPUT_PATH
        else:
            self.output_path = output_path
        self.system_prompt = system_prompt if system_prompt else SYSTEM_PROMPT
        self.chunk0_query = chunk0_query if chunk0_query else CHUNK0_QUERY
        self.chunk_query = chunk_query if chunk_query else CHUNK_QUERY

        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
                print(f"Directory created: {self.output_path}")
            except OSError as e:
                print(f"Error creating directory {self.output_path}: {e}")
        return

    def continue_summarise_pdf(self, pdf_path, chunk_size=8000, temperature=0.8, prev_chunk=0):
        """This method is for handling large files (100s of chunks) which can often resulting in 
        LLM timeout if using summarise_pdf() method.
        It is similar to the summarise_pdf method, with additional capability to allow
        continuing from a previous attempt. It will append to the output file from prev_chunk, 
        so that it can handle errors/disruptions from previous runs where prev_chunk was not completed.
        If prev_chunk=0 then it functions similar to summarise_pdf(), except that it will
        write/append to the output file every 10 chunks (instead of writing only once at the end).
        """
        interval=10 # write to file every interval chunks

        output_path = utils.get_basename_without_extension(pdf_path)
        output_path = f"{self.output_path}/{output_path}.txt"
        if os.path.exists(output_path) and prev_chunk>0:
            f = open(output_path, mode="a", encoding="utf-8")
        else:
            f = open(output_path, mode="w", encoding='utf-8')

        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        chunks = splitter.split_text(text)
        table_name = str(pdf_path).split('-')[1]
        print(f"summarise_pdf got {len(chunks)} chunks, table / doc name: {table_name}")
        print(f"skipping chunks [0, {prev_chunk}).")

        summaries = []
        for j, chunk in enumerate(chunks[prev_chunk:]):
            i = j+prev_chunk
            if i==0:
                queries=[self.chunk0_query]
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"}
                ]
            else:
                queries=[f"this is chunk#{i}" + self.chunk_query]
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"},
                    {"role": "user", "content": f"chunk #0 context: table / doc name is {table_name}"}
                ]
            response = client.chat.completions.create(
                model=utils.CHAT_MODEL,
                messages=messages,
                temperature=temperature,
                # max_tokens=1000
            ) 
        
            summaries.append(response.choices[0].message.content)
            if j % interval == 0:
                f.write('\n\n'.join([str(s) for s in summaries]))
                summaries = []
                print(f"...written chunks up to {i} into file successfully.")

        # write the rest of the chunks into file:
        f.write('\n\n'.join([str(s) for s in summaries]))
        print(f"...written chunks up to {i} into file successfully. The End.")
        return
    
    def summarise_pdf(self, pdf_path, chunk_size=8000, temperature=0.8):
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        # splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=100)
        chunks = splitter.split_text(text)
        table_name = str(pdf_path).split('-')[1]
        print(f"summarise_pdf got {len(chunks)} chunks, table / doc name: {table_name}")

        summaries = []
        for i, chunk in enumerate(chunks):
            if i==0:
                queries=[self.chunk0_query]
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"}
                ]
            else:
                queries=[f"this is chunk#{i}" + self.chunk_query]
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"},
                    {"role": "user", "content": f"chunk #0 context: table / doc name is {table_name}"}
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
            summary = self.summarise_pdf(file_path)
            with open(output_path, mode="w", encoding='utf-8') as f:
                f.write(summary)

        return
        
    def summarise_html_content(self, html_content, chunk_size=8000, temperature=0.8):
        # html_content = utils.unicode_escape_if_outside_utf8(html_content)
        table_name = html_content.split('<title>')[1].split(' -')[0]
        splitter = HTMLSemanticPreservingSplitter(headers_to_split_on=['h1', 'h2', 'h3', 'h4'],
                                                  max_chunk_size=chunk_size, 
                                                  chunk_overlap=100,
                                                  separators=['div', 'tr', 'td'],
                                                  elements_to_preserve=['td'])
        chunks = splitter.split_text(html_content)
        
        print(f"summarise_html, got {len(chunks)} chunks, table / doc name: {table_name}")
        
        summaries = []
        
        for i, chunk in enumerate(chunks):
            if i==0:
                queries=[self.chunk0_query]
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"}
                ]
            else:
                queries=[f"this is chunk#{i}"+self.chunk_query]
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Document Excerpt: {chunk}"},
                    {"role": "user", "content": f"Question: {queries}"},
                    {"role": "user", "content": f"chunk #0 context: table / doc name is {table_name}"}
                ]
            response = client.chat.completions.create(
                model=utils.CHAT_MODEL,
                messages=messages,
                temperature=temperature
                # max_tokens=2048
            ) 
        
            summaries.append(response.choices[0].message.content)

        return '\n\n'.join([str(s) for s in summaries])
    
    def summarise_html(self, html_path, chunk_size=8000, temperature=0.8):
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return self.summarise_html_content(html_content, chunk_size, temperature)
    
    def summarise_html_directory(self, html_directory):
        directory = Path(html_directory)
        for file_path in directory.glob("*.html"):
            output_path = utils.get_basename_without_extension(file_path)
            output_path = f"{self.output_path}/{output_path}.txt"
            # print(file_path)
            summary = self.summarise_html(file_path)
            with open(output_path, mode="w", encoding='utf-8') as f:
                f.write(summary)

        return
    
    def summarise_url(self, url):
        html_content = utils.get_confluence_page(url)
        return self.summarise_html_content(html_content)
