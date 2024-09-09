from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

class ContextEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", verbose=False)

        response_schemas = [
            ResponseSchema(name="is_contextual", description="Whether the file is likely to add context", type="boolean"),
            ResponseSchema(name="reason", description="Reason for the decision", type="string")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        self.evaluation_prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant focused on evaluating file or folder paths to determine if they are likely to add context to a codebase analysis. "
            "Consider the following guidelines:\n"
            "- Files and folders likely to be checked into git are generally important\n"
            "- Exclude files and folders in directories like node_modules, composer dependencies, python modules, build artifacts\n"
            "- Exclude database files (e.g., .sql, .db)\n"
            "- Include source code files, configuration files, and documentation\n\n"
            "Given the path: {path}\n"
            "Type: {path_type} (file or folder)\n"
            "{format_instructions}\n"
            "Provide your evaluation:"
        )
        
        self.evaluation_chain = LLMChain(
            llm=self.llm,
            prompt=self.evaluation_prompt.partial(format_instructions=self.output_parser.get_format_instructions()),
            output_parser=self.output_parser,
            verbose=False
        )
        self.total_evaluations = 0
        self.contextual_count = 0

    def is_contextual(self, path: str, path_type: str) -> tuple[bool, str]:
        result = self.evaluation_chain.run(path=path, path_type=path_type)
        self.total_evaluations += 1
        if result['is_contextual']:
            self.contextual_count += 1
        return result['is_contextual'], result['reason']

    @property
    def contextual_ratio(self):
        return self.contextual_count / self.total_evaluations if self.total_evaluations > 0 else 0