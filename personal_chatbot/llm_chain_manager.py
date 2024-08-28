import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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
    CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID")
    CF_API_KEY = os.environ.get("CF_WORKER_AI_TOKEN")
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
    GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
    GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY")


# LLM provider classes and parameters
LLM_PROVIDERS: Dict[str, Dict[str, object]] = {
    "Cohere": {
        "class": ChatCohere,
        "params": {
            "base_url": Configuration.COHERE_BASE_URL,
            "cohere_api_key": Configuration.COHERE_API_KEY,
        },
    },
    "Anthropic-Haiku-3": {
        "class": ChatAnthropic,
        "params": {
            "model_name": "claude-3-haiku-20240307",
            "api_key": Configuration.ANTHROPIC_API_KEY,
        },
    },
    "Anthropic-Sonnet-3.5": {
        "class": ChatAnthropic,
        "params": {
            "model_name": "claude-3-5-sonnet-20240620",
            "api_key": Configuration.ANTHROPIC_API_KEY,
        },
    },
    "Anthropic-Opus-3": {
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
    "Google-Gemini-1.5-flash-latest": {
        "class": ChatGoogleGenerativeAI,
        "params": {
            "model": "gemini-1.5-flash-latest",
            "api_key": Configuration.GOOGLE_API_KEY,
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
    "Cloudflare-llama-3": {
        "class": CloudflareWorkersAI,
        "params": {
            "account_id": Configuration.CF_ACCOUNT_ID,
            "api_token": Configuration.CF_API_KEY,
            "model": "@cf/meta/llama-3-8b-instruct",
        },
        "use_proxy": False,
    },
    "NVIDIA-llama-3.1": {
        "class": ChatOpenAI,
        "params": {
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": Configuration.NVIDIA_API_KEY,
            "model": "meta/llama-3.1-405b-instruct",
        },
        "use_proxy": False,
    },
}


# LLMChain Logic Manager
class LLMChainManager:
    """
    Manager class for LLMChain logic. Initializes and manages the LLMChain components.
    """

    def __init__(self, system_prompt, temperature, use_tools=False):
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.use_tools = use_tools
        self.llm = None
        self.prompt = None
        self.llm_chain = None
        self.tools = [
            Tool(
                name="DuckDuckGo_Search",
                func=DuckDuckGoSearchRun(),
                description="DuckDuckGo tool to search the web for information.",
            ),
            Tool(
                name="Google_Search",
                description="Google tool to search the web for information.",
                func=GoogleSearchAPIWrapper(
                    google_cse_id=Configuration.GOOGLE_CSE_ID,
                    google_api_key=Configuration.GOOGLE_SEARCH_API_KEY,
                ).run,
            ),
        ]

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
            if proxy:
                os.environ["http_proxy"] = proxy
                os.environ["HTTP_PROXY"] = proxy
                os.environ["https_proxy"] = proxy
                os.environ["HTTPS_PROXY"] = proxy

        self.llm = llm_class(**llm_params)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def init_prompt(self):
        """
        Initialize the prompt template with the system prompt and placeholders for
        chat history and human input.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

    def init_llm_chain(self):
        """
        Initialize the LLMChain with the LLM, prompt, and memory components.
        """
        if self.use_tools:
            agent = create_tool_calling_agent(
                self.llm_with_tools, self.tools, self.prompt
            )
            self.llm_chain = AgentExecutor(agent=agent, tools=self.tools)
        else:
            self.llm_chain = self.prompt | self.llm | StrOutputParser()
