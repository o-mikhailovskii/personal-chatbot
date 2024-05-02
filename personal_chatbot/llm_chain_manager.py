import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


# Load the necessary environment variables
@dataclass
class Configuration:
    """
    Dataclass to store LLM API credentials and base URL.
    """

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    COHERE_BASE_URL = os.getenv("COHERE_BASE_URL")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# LLM provider classes and parameters
LLM_PROVIDERS: Dict[str, Dict[str, object]] = {
    "Cohere": {
        "class": ChatCohere,
        "params": {
            "base_url": Configuration.COHERE_BASE_URL,
            "cohere_api_key": Configuration.COHERE_API_KEY,
        },
    },
    "Anthropic-Haiku": {
        "class": ChatAnthropic,
        "params": {
            "model_name": "claude-3-haiku-20240307",
            "api_key": Configuration.ANTHROPIC_API_KEY,
        },
    },
    "Anthropic-Sonnet": {
        "class": ChatAnthropic,
        "params": {
            "model_name": "claude-3-sonnet-20240229",
            "api_key": Configuration.ANTHROPIC_API_KEY,
        },
    },
    "Anthropic-Opus": {
        "class": ChatAnthropic,
        "params": {
            "model_name": "claude-3-opus-20240229",
            "api_key": Configuration.ANTHROPIC_API_KEY,
        },
    },
    "Google-Gemini-1.5-pro-latest": {
        "class": ChatGoogleGenerativeAI,
        "params": {
            "model": "gemini-1.5-pro-latest",
            "api_key": Configuration.GOOGLE_API_KEY,
            # "convert_system_message_to_human": True,
        },
        "use_proxy": True,
    },
    "Groq-llama3-70b-8192": {
        "class": ChatGroq,
        "params": {
            "model_name": "llama3-70b-8192",
            "groq_api_key": Configuration.GROQ_API_KEY,
        },
    },
    "Groq-mixtral-8x7b-32768": {
        "class": ChatGroq,
        "params": {
            "model_name": "mixtral-8x7b-32768",
            "groq_api_key": Configuration.GROQ_API_KEY,
        },
    },
    "Ollama-phi3": {
        "class": ChatOllama,
        "params": {
            "model": "phi3",
        },
        "use_proxy": False,
    },
}


# LLMChain Logic Manager
class LLMChainManager:
    """
    Manager class for LLMChain logic. Initializes and manages the LLMChain components.
    """

    def __init__(self, system_prompt, temperature):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.llm = None
        self.prompt = None
        self.memory = None
        self.llm_chain = None

    def init_llm(self, provider):
        """
        Initialize the LLM component based on the provider.
        """
        provider_config = LLM_PROVIDERS.get(provider)
        if not provider_config:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        llm_class = provider_config["class"]
        llm_params = provider_config["params"]
        use_proxy = provider_config.get("use_proxy", False)
        if use_proxy:
            proxy = os.getenv("PROXY")
            os.environ["http_proxy"] = proxy
            os.environ["HTTP_PROXY"] = proxy
            os.environ["https_proxy"] = proxy
            os.environ["HTTPS_PROXY"] = proxy

        self.llm = llm_class(**llm_params)

    def init_prompt(self):
        """
        Initialize the prompt template with the system prompt and placeholders for
        chat history and human input.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

    def init_memory(self):
        """
        Initialize the memory component to store chat history.
        """
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def init_llm_chain(self):
        """
        Initialize the LLMChain with the LLM, prompt, and memory components.
        """
        self.llm_chain = LLMChain(
            llm=self.llm, prompt=self.prompt, memory=self.memory, verbose=True
        )
