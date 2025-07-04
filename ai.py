from pathlib import Path

from langchain.chat_models.base import init_chat_model
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate

prompt_dir = Path(__file__).parent / "prompts"
system_prompt_file = prompt_dir / "system.txt"
user_prompt_file = prompt_dir / "user.txt"


async def query() -> str:
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(
                prompt=PromptTemplate.from_file(system_prompt_file)
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate.from_file(user_prompt_file)
            ),
        ]
    )

    char_model = init_chat_model(
        model="deepseek-r1:14b",
        model_provider="openai",
        base_url="http://192.168.11.124:8750/v1",
        api_key="abc",
        temperature=0,
    )

    model = prompt_template | char_model | JsonOutputParser()
    await model.ainvoke({})
