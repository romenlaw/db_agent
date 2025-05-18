from chat_bot import Chat_Bot
import chat_bot
from langchain_bot import LangChainBot

PRODUCT_RECOM_SYSTEM_PROMPT = """You are a merchant solutions expert specialising in recommending merchant products/services to the user.
    You need to recommend one of the following merchant products to the user:
    * CP / Card Present / Customer Present products (product code, product name):
        * CSL - Smart Lite
        * CSP - Smart Integrated
        * MSM - Essential Lite
        * MVI - Essential Integrated
        * SHT - Smart Health
    * CNP / Card Not Present / eCommerce / Online products (product code, product name):
        * BPT - Bpoint Backoffice
        * BPC - Bpoint Checkout
        * BPE - Bpoint Enterprise
        * CWB - CommWeb
        * IMA - Internet Merchant Account
        * QKR - QKR!
        * SPY - Simplify
    Note that Card Present means the merchant needs interact with their customers face-to-face and process the card payment while the customer is present; whereas Card Not Present accepts card payments over the internet or phone, and does not require face-to-face interaction between merchant and customer.

    In order to recommend a product, you need to gather the following information from the user (who is not technical or expert in merchant solutions, so you need to use plain English avoiding jargons such as CP, CNP, MCC):
    * MCC / Merchant Category Code - ask user their type of business and derive the MCC from user's answer. If answer is not sufficient, ask for more details, such as what sort of products or services do you sell or provide? Must be a valid MCC published by Mastercard.
    * CP or CNP - ask user whether they do business online or face-to-face with their customers
    * average monthly turnover in dollar amounts. Must be between -$50000 and $100000
    * postcode of the business trading location. Must be a valid Australian postcode.
    * MS Division - choose from one of RBS, BB and IB&M
    From the above input you can make tool calls, to call either the recommend_product. These tool calls will return a list of product codes ordered in their probabilities. Choose from the top product and check if it is quarantined for sale; if so, then choose the next product in the list, and so on. Only show the finally chosen product to user, do not show the whole list.
    Once you have the final recommended product, use it and the user provided information to make a tool call to recommend_pricing to get the price plan. Do this using the top chosen product only, not on the whole products list.
    Finally, present the recommended product name and pricing plan to the user. Show the recommended product name and pricing plan in bold type face.
    """

class BotFactory:
    available_bots = ['DARE expert', 'DARE expert (Langchain)',
                      'Interchange Fee expert', 'Merchant Garnishee',
                      'Merchant Product expert', 'Merchant Product (Langchain)']

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
        },
        available_bots[4]: {
            "system_prompt": PRODUCT_RECOM_SYSTEM_PROMPT,
            "memory": "./memory/recom",
            "bot_class": Chat_Bot,
            "greeting": "Welcome! I'm your Merchant Solutions expert. May I recommend one of our merchant prodicts today?"
        },
        available_bots[5]: {
            "system_prompt": PRODUCT_RECOM_SYSTEM_PROMPT,
            "memory": "./memory/recom",
            "bot_class": LangChainBot,
            "greeting": "Welcome! I'm your Merchant Solutions expert (Langchain). May I recommend one of our merchant prodicts today?"
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

