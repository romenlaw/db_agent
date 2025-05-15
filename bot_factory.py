from chat_bot import Chat_Bot
import chat_bot
from langchain_bot import LangChainBot

class BotFactory:
    available_bots = ['DARE expert', 'DARE expert (Langchain)',
                      'Interchange Fee expert', 'Merchant Garnishee']

    _config = {
        available_bots[0]: {
            "system_prompt": chat_bot.SYSTEM_PROMPT,
            "memory": './memory/dare',
            "bot_class": Chat_Bot,
            "greeting": "Welcome! I'm your DARE database assistant. How can I help you today?",
        },
        available_bots[1]: {
            "system_prompt": chat_bot.SYSTEM_PROMPT,
            "memory": './memory/dare_lc',
            "bot_class": LangChainBot,
            "greeting": "Welcome! I'm your DARE database assistant (Langchain). How can I help you today?",
        },
        available_bots[2]: {
            "system_prompt": """
    You are an expert in schemes (Mastercard / MC, Visa, eftpos) interchange fees and programs.
    Be a helpful assistant to the user in answering their questions.
    """,
            "memory": "./memory/scheme",
            "bot_class": Chat_Bot,
            "greeting": "Welcome! I'm your Interchange Fee expert. How can I help you today?"
        },
        available_bots[3]: {
            "system_prompt": """You are a knowledgeable assistant. Use the provided context as the primary source of information to answer the query. 
    If the context is insufficient or lacks details, supplement it with your general knowledge to provide a complete and accurate response. 
    Clearly prioritize the provided context when it applies. Quote the provided context to support your answers.
    Maintain conversational context from the chat history when relevant.
    """,
            "memory": "./memory/garnishee",
            "bot_class": Chat_Bot,
            "greeting": "Welcome! I'm your Merchant Garnishee Tool expert. How can I help you today?"
        }
    }

    @staticmethod
    def bot(bot=available_bots[0]):
        print("BotFactory: creating ", bot)
        system_prompt = BotFactory._config[bot]['system_prompt']
        memory_path = BotFactory._config[bot]['memory']
        bot_class = BotFactory._config[bot]['bot_class']
        cb = bot_class(memory_path=memory_path, system_prompt=system_prompt)
        return cb

