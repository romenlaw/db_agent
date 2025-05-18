from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from itertools import islice
from chat_bot import Chat_Bot
import recommend
import chat_bot
import utils
import db_utils
import json

MEMORY_PATH = './memory_lc'

def get_first_n_chunks(vector_store, n=3):
    # Access index_to_docstore_id mapping
    index_to_id = vector_store.index_to_docstore_id
    # Get first n items from the mapping
    first_n_ids = list(islice(index_to_id.items(), n))
    # Retrieve documents from docstore
    chunks = []
    for index, doc_id in first_n_ids:
        doc = vector_store.docstore._dict.get(doc_id)
        if doc:
            # print("doc=", doc)
            # chunks.append((index, doc.page_content, doc.metadata))
            chunks.append(doc)
    return chunks

class LangChainBot(Chat_Bot):
    def __init__(self, memory_path=MEMORY_PATH, emb_model='text-embedding-3-large_v1',
                 system_prompt=None):
        self.system_prompt = system_prompt if system_prompt else chat_bot.SYSTEM_PROMPT
        self.memory_path = memory_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=utils.GENAI_API_KEY, 
                                           base_url=utils.GENAI_API_URL,
                                           model=emb_model)
        self.chat_history = []
        self.temperature = 0.3
        self.model_id = utils.CHAT_MODEL

        self.load_memory()
        self.setup_agent()
        
    def load_memory(self):
        """Load or create FAISS vector store"""
        try:
            self.vectorstore = FAISS.load_local(
                self.memory_path + "/faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            # If no existing index, create one from documents
            print("No existing FAISS index found. Please create one first.")
            raise

    def setup_agent(self):
        """Setup LangChain agent with tools"""
        # Define the SQL execution tool
        self.tools = [
            Tool(
                name="get_current_date_time",
                func=lambda query: json.dumps(utils.print_now()),
                description="Get the current date, time and day of week. It answers the question of today's date, day of week, etc."
            ),
            Tool(
                name="execute_sql",
                func=lambda query: json.dumps(db_utils.execute_sql(query)),
                description="Execute SQL query on the database and return results. The query parameter should be a valid SQL query string."
            ),
            recommend.recommend_product_wrapper,
            recommend.recommend_pricing_wrapper
        ]

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        llm = ChatOpenAI(
            temperature=self.temperature,
            model=self.model_id,
            openai_api_key=utils.GENAI_API_KEY,
            base_url=utils.GENAI_API_URL
        )
        agent = create_openai_tools_agent(llm, self.tools, self.prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True
        )

    def search_chunks(self, query):
        """Search for relevant document chunks"""
        docs = self.vectorstore.similarity_search(query, k=5)
        docs.extend(get_first_n_chunks(self.vectorstore, 3))

        return "\n===\n".join([doc.page_content for doc in docs])

    def chat(self, prompt, model=utils.CHAT_MODEL, temperature=0.3):
        """
        Chat with the bot using LangChain components
        
        Args:
            prompt (str): user prompt or question
            model (str): LLM model id
            temperature (float): ranging from 0.0 to 2.0
        """
        # Get relevant context
        context = self.search_chunks(prompt)
        
        # Format input with context
        full_prompt = f"Document Excerpt: {context}\nQuestion: {prompt}"
        
        # Create new LLM if temperature or model changed
        if temperature != self.temperature or model != self.model_id:
            self.setup_agent()
        
        # Initial agent invocation
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add chat history
        if self.chat_history:
            messages.extend([
                {"role": msg.type, "content": msg.content}
                for msg in self.chat_history
            ])
        
        # Add current context and question
        messages.extend([
            {"role": "user", "content": full_prompt}
        ])
        
        # Initial agent run
        response = self._run_agent_with_tools(messages)
        
        # Update chat history
        self.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=response)
        ])
        
        # Keep only last 20 message pairs to prevent context from growing too large
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
            
        return response

    def _run_agent_with_tools(self, messages):
        """
        Run agent with recursive tool handling
        
        Args:
            agent: The LangChain agent
            messages: List of conversation messages
        """
        # Invoke agent
        result = self.agent_executor.invoke({
            "input": messages[-1]["content"],
            "chat_history": messages[:-1]  # Exclude current message
        })
        # print("result keys=", *result.keys(), sep=",")
        return result["output"]
        

if __name__ == "__main__":
    print('Initializing LangChain bot, please wait...')
    bot = LangChainBot()
    print(bot.chat('what is DARE', temperature=0.8))
