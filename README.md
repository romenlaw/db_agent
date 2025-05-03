# Overview
db_agent is a RAG agent that has knowledge about database data dictionary. This knowledge can be used to generate SQL queries and execute them in the database.
The general flow follows the generic RAG workflow:
![RAG overview](https://www.kdnuggets.com/wp-content/uploads/Ferrer_RAG_LlamaIndex_2.png)

# Installation
1. clone the repo
2. install the python dependencies. TODO: create requirement.txt with cline
3. set up environment variables GENAI_API_KEY, GENAI_API_URL to suit your corporate envrionment
4. to connect to databse, setup your ODBC connection to the database server

# Create Memory
Follow the steps in scheme_tester.ipynb:
1. save your knowledgebase contents as PDF or HTML in an input directory
2. use LLM to convert the input files into Markdown format and save them in another directory
3. embed the Markdown text files and save them into a memory directory

# Chat
to try the app:
```
python chat_bot_gui.py
```
