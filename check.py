import langchain
from langchain_community.chat_message_histories import ChatMessageHistory

print("--- LangChain Check ---")
print(f"LangChain Version: {langchain.__version__}")
print(f"LangChain Core Version: (checking community package)")
print("Chat message history imported successfully!")
print("-----------------------")