# Structured Outputs
LLMs producing structured outputs unlock downstream applications that rely on reliable parsing rules. LlamaIndex depends on structured responses for several key workflows.

- **Document retrieval:** Internal index structures expect outputs to follow a schema (for example, the tree index looks for answers formatted as `ANSWER: (number)`).
- **Response synthesis:** User-facing responses often include structured components such as JSON objects, tables, or SQL snippets.

LlamaIndex provides helper modules built on the default LLM classes to keep outputs predictable:

- **Pydantic Programs:** General-purpose programs that map prompt inputs to Pydantic objects. These can use function-calling APIs or text completions with output parsers.
- **Pre-defined Pydantic Programs:** Curated programs that target specific output types like dataframes.
- **Output Parsers:** Middleware that adds formatting instructions before an LLM call and parses the returned text after it completes. Function-calling LLMs already return structured payloads, so parsers are optional in those cases.

## Anatomy of a Structured Output Function

The pipeline depends on whether you call a generic completion API or a function-calling endpoint.

### Generic completion APIs

Inputs and outputs flow through plain-text prompts. The output parser injects formatting guidance before the call and normalizes the returned text afterward.

### Function-calling APIs

Function-calling endpoints already respond with structured payloads; the parser simply casts those payloads into the expected object (for example, a Pydantic model).

## Starter Guides

- Structured data extraction tutorial
- Examples of Structured Outputs
- Other Resources
- Pydantic Programs
- Structured Outputs + Query Engines
- Output Parsers

# Output Parsing Modules
LlamaIndex integrates with output parsing modules provided by other frameworks. These modules can feed formatting instructions to any prompt/query via `output_parser.format` and parse responses with `output_parser.parse`.

## Guardrails
Guardrails is an open-source Python package for specification, validation, and correction of output schemas. Below is an example integrating it with LlamaIndex.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.output_parsers.guardrails import GuardrailsOutputParser
from llama_index.llms.openai import OpenAI

# load documents, build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex(documents, chunk_size=512)

# define query / output spec
rail_spec = """
<rail version="0.1">

<output>
    <list name="points" description="Bullet points regarding events in the author's life.">
        <object>
            <string name="explanation" format="one-line" on-fail-one-line="noop" />
            <string name="explanation2" format="one-line" on-fail-one-line="noop" />
            <string name="explanation3" format="one-line" on-fail-one-line="noop" />
        </object>
    </list>
</output>

<prompt>

Query string here.

@xml_prefix_prompt

{output_schema}

@json_suffix_prompt_v2_wo_none
</prompt>
</rail>
"""

# define output parser
output_parser = GuardrailsOutputParser.from_rail_string(
    rail_spec, llm=OpenAI()
)

# attach output parser to LLM
llm = OpenAI(output_parser=output_parser)

# obtain a structured response
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query(
    "What are the three items the author did growing up?",
)
print(response)
```

Output:

```json
{
    "points": [
        {
            "explanation": "Writing short stories",
            "explanation2": "Programming on an IBM 1401",
            "explanation3": "Using microcomputers"
        }
    ]
}
```

## Langchain
Langchain also exposes structured output parsers that can be used from LlamaIndex.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.llms.openai import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# load documents, build index
documents = SimpleDirectoryReader("../paul_graham_essay/data").load_data()
index = VectorStoreIndex.from_documents(documents)

# define output schema
response_schemas = [
    ResponseSchema(
        name="Education",
        description="Describes the author's educational experience/background.",
    ),
    ResponseSchema(
        name="Work",
        description="Describes the author's work experience/background.",
    ),
]

# define output parser
lc_output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)
output_parser = LangchainOutputParser(lc_output_parser)

# attach output parser to LLM
llm = OpenAI(output_parser=output_parser)

# obtain a structured response
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query(
    "What are a few things the author did growing up?",
)
print(str(response))
```

Output:

```json
{
    "Education": "Before college, the author wrote short stories and experimented with programming on an IBM 1401.",
    "Work": "The author worked on writing and programming outside of school."
}
```

## Guides
More examples:

- Guardrails
- Langchain
- Guidance Pydantic Program
- Guidance Sub-Question
- Openai Pydantic Program
