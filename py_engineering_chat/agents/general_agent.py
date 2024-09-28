from pydantic import BaseModel, Field
from langchain.graphs import StateGraph
from langchain.graphs.state_graph import START, END
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import render_text_description
from langchain.prompts import MessagesPlaceholder
from py_engineering_chat.util.tiered_memory import TieredMemory
from py_engineering_chat.util.memory_refiner import MemoryRefiner
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager
from py_engineering_chat.util.command_parser import parse_commands
from py_engineering_chat.tools.base_tool import BaseProjectTool
from py_engineering_chat.tools.file_tool import FileTool
from py_engineering_chat.tools.git_tool import GitTool
from py_engineering_chat.tools.search_tool import SearchTool
from py_engineering_chat.tools.summarize_tool import SummarizeTool
from py_engineering_chat.tools.research_tool import ResearchTool
from py_engineering_chat.tools.code_analysis_tool import CodeAnalysisTool
from py_engineering_chat.tools.code_generation_tool import CodeGenerationTool
from py_engineering_chat.tools.code_modification_tool import CodeModificationTool
from py_engineering_chat.tools.test_generation_tool import TestGenerationTool
from py_engineering_chat.tools.documentation_tool import DocumentationTool
from py_engineering_chat.tools.dependency_management_tool import DependencyManagementTool
from py_engineering_chat.tools.project_management_tool import ProjectManagementTool
from py_engineering_chat.tools.deployment_tool import DeploymentTool
from py_engineering_chat.tools.database_tool import DatabaseTool
from py_engineering_chat.tools.api_tool import APITool
from py_engineering_chat.tools.security_tool import SecurityTool
from py_engineering_chat.tools.performance_tool import PerformanceTool
from py_engineering_chat.tools.logging_tool import LoggingTool
from py_engineering_chat.tools.monitoring_tool import MonitoringTool
from py_engineering_chat.tools.ci_cd_tool import CICDTool
from py_engineering_chat.tools.containerization_tool import ContainerizationTool
from py_engineering_chat.tools.cloud_tool import CloudTool
from sentence_transformers import SentenceTransformer
import os
import time

class State(BaseModel):
    messages: list = Field(default_factory=list)
    next: str = START

class GeneralAgent:
    def __init__(self):
        self.graph_builder = StateGraph(State)
        self.tiered_memory = TieredMemory()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = get_configured_logger(__name__)
        self.edit_mode = False  # Default to read-only mode
        self.memory_refiner = MemoryRefiner(
            self.tiered_memory,
            ChatOpenAI(temperature=0, model_name="gpt-4"),
            refinement_interval=300  # 5 minutes
        )

        # Initialize tools
        self.tools = [
            FileTool(),
            GitTool(),
            SearchTool(),
            SummarizeTool(),
            ResearchTool(),
            CodeAnalysisTool(),
            CodeGenerationTool(),
            CodeModificationTool(),
            TestGenerationTool(),
            DocumentationTool(),
            DependencyManagementTool(),
            ProjectManagementTool(),
            DeploymentTool(),
            DatabaseTool(),
            APITool(),
            SecurityTool(),
            PerformanceTool(),
            LoggingTool(),
            MonitoringTool(),
            CICDTool(),
            ContainerizationTool(),
            CloudTool(),
        ]

        # Initialize the agent
        self.agent = self._create_agent()

        # Build the graph
        self._build_graph()

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        llm = ChatOpenAI(temperature=0, model="gpt-4")

        agent = create_openai_functions_agent(llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=ConversationBufferMemory(return_messages=True),
        )

    def _build_graph(self):
        self.graph_builder.add_node("agent", self._run_agent)
        self.graph_builder.add_edge(START, "agent")
        self.graph_builder.add_edge("agent", END)

    def _run_agent(self, state):
        result = self.agent.invoke(state)
        return {"messages": [HumanMessage(content=state["messages"][-1]), AIMessage(content=result["output"])]}

    async def run_conversation(self):
        graph = self.graph_builder.compile()
        
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            context_data_list = parse_commands(user_input, ChatSettingsManager())
            context_strings = [context_data.toString() for context_data in context_data_list]
            context = " ".join(context_strings)

            state = {"messages": [user_input]}
            for output in graph.stream(state):
                if output.get("messages"):
                    print("Assistant:", output["messages"][-1].content)

            # Refine memory periodically
            self.memory_refiner.maybe_refine_memory()

async def run_continuous_conversation():
    agent = GeneralAgent()
    await agent.run_conversation()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_continuous_conversation())
