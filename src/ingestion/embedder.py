from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import load_config
from src.utils.env_utils import get_env_variable



from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import load_config
from src.utils.env_utils import get_env_variable


def get_embedding_model():

    config = load_config()

    provider = config.embedding.provider

    if provider == "openai":

        api_key = get_env_variable("OPENAI_API_KEY")

        embeddings = OpenAIEmbeddings(
            model=config.embedding.openai.model,
            api_key=api_key
        )

    elif provider == "gemini":

        api_key = get_env_variable("GOOGLE_API_KEY")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding.gemini.model,
            google_api_key=api_key
        )

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    return embeddings
