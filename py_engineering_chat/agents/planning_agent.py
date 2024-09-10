from typing import Dict, Any, List, Literal, Optional
import json
import os
from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager
from py_engineering_chat.tools.shell_command_tool import SafeShellCommandTool
import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits import FileManagementToolkit

class PlanStep(BaseModel):
    step_number: int
    description: str
    file_changes: List[Dict[str, str]]
    types_involved: List[str]
    new_methods: List[Dict[str, str]]
    test_plan: str

class Plan(BaseModel):
    task_title: str
    description: str
    steps: List[PlanStep]

class PlanningAction(BaseModel):
    """Action to take in the planning process."""
    action: Literal["create_plan", "refine_plan", "ask_question", "finalize_plan"] = Field(
        ...,
        description="The action to take in the planning process.",
    )
    plan: Plan = Field(
        ...,
        description="The current state of the plan.",
    )
    question: Optional[str] = Field(
        None,
        description="A question to ask for more information, if needed.",
    )
    is_satisfied: bool = Field(
        False,
        description="Whether the planner is satisfied with the current plan.",
    )

class PlanningAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
        self.plan = Plan(task_title="", description="", steps=[])
        self.settings_manager = ChatSettingsManager()
        self.structured_planner = self.model.with_structured_output(PlanningAction)
        self.is_windows = os.getenv('IS_WINDOWS', 'false').lower() == 'true'
        self.ai_shadow_directory = self.create_ai_shadow_directory()
        self.tools = self.get_tools()
        self.agent_executor = self.create_chain()

    def create_ai_shadow_directory(self):
        ai_shadow_dir = os.getenv('AI_SHADOW_DIRECTORY', './ai_shadow')
        if self.is_windows:
            ai_shadow_dir = ai_shadow_dir.replace('/', '\\')
        os.makedirs(ai_shadow_dir, exist_ok=True)
        return ai_shadow_dir

    def get_file_tools(self):
        toolkit = FileManagementToolkit(root_dir=self.ai_shadow_directory)
        return toolkit.get_tools()

    def get_tools(self):
        file_tools = self.get_file_tools()
        custom_tools = [SafeShellCommandTool()]
        return file_tools + custom_tools

    def create_chain(self):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a planning assistant with file management capabilities. Create, refine, and evaluate a detailed plan based on the following input. The plan should follow the structure defined by the Plan model. Continuously improve the plan and ask questions if more information is needed. Only finalize the plan when you are satisfied it is comprehensive and of high quality. The current system is {'Windows' if self.is_windows else 'not Windows'}. Always use the following format:\n\nThought: your thoughts here\nAction: the action to take, should be one of {{tool_names}}\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question"),
            ("human", "Current plan: {current_plan}\n\nTask: {input}\n\nContext: {context}\n\ntool names: {tool_names}\n\nAction Input: {tools}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Determine the next action: create_plan, refine_plan, ask_question, or finalize_plan. Update the plan accordingly and set is_satisfied to True only when the plan is complete and of high quality."),
            ("human", "Thought: {agent_scratchpad}"),
        ])

        agent = create_react_agent(self.model, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, memory=memory, verbose=True)

    def generate_and_refine_plan(self, task_title: str, description: str) -> Plan:
        initial_prompt = f"Task Title: {task_title}\nDescription: {description}\n\nCreate a detailed plan with granular steps. Each step should describe file changes, types involved, new methods, inputs, and outputs. Also, include a test plan for each step."
        
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while iteration < max_iterations:
            processed_input = self.process_input({"input": initial_prompt})
            response = self.agent_executor.invoke({
                "input": processed_input["input"],
                "current_plan": json.dumps(self.plan.model_dump(), default=str),
                "context": processed_input["context"],
                "tool_names": ", ".join([tool.name for tool in self.tools]),
                "tools": str(self.tools)
            })
            
            if response.action == "ask_question":
                print(f"\nQuestion from the Planning Agent: {response.question}")
                answer = input("Your answer: ")
                initial_prompt = f"Question: {response.question}\nAnswer: {answer}\n\nContinue refining the plan with this information."
            elif response.action == "finalize_plan" and response.is_satisfied:
                return response.plan
            else:
                self.plan = response.plan
                print("\nCurrent Plan State:")
                print(json.dumps(self.plan.model_dump(), indent=2))
                feedback = input("\nProvide feedback on the current plan (or press Enter to continue): ")
                if feedback:
                    initial_prompt = f"User feedback: {feedback}\n\nContinue refining and improving the plan based on this feedback."
                else:
                    initial_prompt = "Continue refining and improving the plan. Ask questions if needed."
            
            iteration += 1
        
        print("Maximum iterations reached. Finalizing the current plan.")
        return self.plan

    def save_plan(self) -> None:
        plan_path = self.shadow_path / "project_plan.json"
        with open(plan_path, "w") as f:
            json.dump(self.plan.model_dump(), f, indent=2)

    def chat(self):
        print("Welcome to the Planning Agent. Let's create a detailed project plan.")
        
        while True:
            project_name = input("Enter the project name: ")
            if self.settings_manager.get_setting(f'projects.{project_name}'):
                self.settings_manager.set_setting('current_project', project_name)
                print(f"Current project set to: {project_name}")
                break
            else:
                print(f"Project '{project_name}' not found in chat settings. Please try again.")

        task_title = input("Enter the task title: ")
        description = input("Enter the task description: ")

        print("\nGenerating and refining the plan. This process will be interactive.")
        print("You can use @docs:<collection_name> or @codebase:<query> to reference additional context.")
        self.plan = self.generate_and_refine_plan(task_title, description)
        
        print("\nFinal Plan:")
        print(json.dumps(self.plan.model_dump(), indent=2))

        self.save_plan()
        print(f"\nPlan saved to {self.shadow_path / 'project_plan.json'}")

def chat_with_planning_agent():
    agent = PlanningAgent()
    agent.chat()

if __name__ == "__main__":
    chat_with_planning_agent()