import faiss
import numpy as np
import json
import utils
import db_utils
# from db_utils import DbUtil
from embedder import Embedder

# def execute_sql(query):
#     """Execute SQL query on the database and return results

#     Args:
#         query: SQL query to be executed
#     """
#     try:
#         db = DbUtil()
#         df = db.execute(query)
#         db.clean_up()

#         return {"status": "success", "results": str(df)}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}


MEMORY_PATH = './memory'
system_prompt = """
    You are a SQL Server expert speicalising in DARE database and SQL queries in T-SQL. 
    Use the provided context as the primary source of information to answer the query. 
    Do not make assumptions outside of the provided source.
    Maintain conversational context from the chat history when relevant.
    When generating SQL statements explain your chain of thoughts. 
    Always limit your SELECT queries using SELECT TOP (1000) ...
    """

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute SQL query on the database and return results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

class Chat_Bot():
    def __init__(self, memory_path=MEMORY_PATH, emb_model=None):
        self.memory_path = memory_path
        self.load_memory()
        self.chat_history = []
        self.embedder = Embedder(model=emb_model) if emb_model else Embedder()
      

    def load_memory(self):
        self.embeddings = np.load(f'{self.memory_path}/embeddings.npy')
        with open(f'{self.memory_path}/chunks.txt', 'r', encoding='utf-8') as f:
            self.chunks = f.read().split('\n===\n')
        self.indices = faiss.read_index(f'{self.memory_path}/faiss_index.bin')
        
    def search_chunks(self, queries):
        """queries - list of questions/strings
        """
        # query_embedding = np.array(get_embedding(query)).reshape(1, -1)
        query_embedding = np.array(self.embedder.embed_chunks(queries))
        distances, indices = self.indices.search(query_embedding, k=5)  # Retrieve top k relevant chunks
        
        # first 3 chunks are the general domain knowledge. Always include them.
        chunks = self.chunks[:3]
        chunks.extend([self.chunks[i] for i in indices[0]])
        return chunks
    
    def new_chat(self):
        """start a new chat, i.e. clear chat history."""
    
    def chat(self, prompt, model=utils.CHAT_MODEL, temperature=0.3):
        """
        Args:
            prompt (str): user prompt or question for the AI bot
            model (str): LLM model id
            temperatur (float): ranging from 0.0 to 2.0, the bigger the wilder
        """
        queries = [prompt]
        relevant_chunks = self.search_chunks(queries)
        
        content = "\n".join(relevant_chunks)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history if available
        if self.chat_history:
            messages.extend(self.chat_history)
        
        # Add current context and question
        messages.extend([
            {"role": "user", "content": f"Document Excerpt: {content}"},
            {"role": "user", "content": f"Question: {prompt}"}
        ])
        
        response = utils.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
            tools=tools, tool_choice='auto'
            # max_tokens=4096
        )
        answer = response.choices[0].message.content
        # answer = ""
        # for word in response.choices[0].message.content:
        #     answer += word
        #     print(word, end="", flush=True)  # Print word with space, flush to show immediately
        #     # time.sleep(0.1)  # Optional: Add delay for typewriter effect
        # print()  # Newline at the end

        # invoke tools
        result = Chat_Bot.process_tool_calls(response)
        final_answer = None
        if result:
            tool_call = response.choices[0].message.tool_calls[0]
            # print(f"tool_call id={tool_call.id}")
            messages = [
                # {"role": "system", "content": system_prompt},
                # {"role": "user", "content": f"Document Excerpt: {content}"},
                {"role": "user", "content": f"Question: {prompt}"},
                {"role": "assistant", "content": str(result), "tool_calls": response.choices[0].message.tool_calls},
                {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
            ]
            
            follow_up_response = utils.client.chat.completions.create(
                model=model,
                messages=messages
            )
            final_answer = follow_up_response.choices[0].message.content
            answer = final_answer

        # Update chat history
        self.chat_history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ])
        
        # Keep only last 10 messages to prevent context from growing too large
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return answer #, final_answer
        
    @staticmethod
    def process_tool_calls(response):
        # Check if the response contains tool calls
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Call the appropriate function
            if function_name == "execute_sql":
                result = db_utils.execute_sql(arguments["query"])
                # print("query result: ", result)
                return result
        return None

if __name__ == "__main__":
    print('please wait, this may take a while to load ...')
    cb = Chat_Bot(memory_path='./memory_html')
    # cb = Chat_Bot(memory_path='./memory_pdf')
    print(cb.chat('what is DARE', temperature=0.8))
