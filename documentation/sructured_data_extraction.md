# Introduction to Structured Data Extraction
LLMs excel at data understanding, leading to one of their most important use cases: the ability to turn regular human language (which we refer to as unstructured data) into specific, regular, expected formats for consumption by computer programs. We call the output of this process structured data. Since in the process of conversion a lot of superfluous data is often ignored, we call it extraction.

The core of the way structured data extraction works in LlamaIndex is Pydantic classes: you define a data structure in Pydantic and LlamaIndex works with Pydantic to coerce the output of the LLM into that structure.

## What is Pydantic?
Pydantic is a widely-used data validation and conversion library. It relies heavily on Python type declarations. There is an extensive guide to Pydantic in that project's documentation, but we'll cover the very basics here.

To create a Pydantic class, inherit from Pydantic's BaseModel class:

from pydantic import BaseModel


class User(BaseModel):
    id: int
    name: str = "Jane Doe"

In this example, you've created a User class with two fields, id and name. You've defined id as an integer, and name as a string that defaults to Jane Doe.

You can create more complex structures by nesting these models:

```python
from typing import List, Optional
from pydantic import BaseModel


class Foo(BaseModel):
    count: int
    size: Optional[float] = None


class Bar(BaseModel):
    apple: str = "x"
    banana: str = "y"


class Spam(BaseModel):
    foo: Foo
    bars: List[Bar]
```

Now Spam has a foo and a bars. Foo has a count and an optional size , and bars is a List of objects each of which has an apple and banana property.

## Converting Pydantic objects to JSON schemas
Pydantic supports converting Pydantic classes into JSON-serialized schema objects which conform to popular standards. The User class above for instance serializes into this:

```json
{
  "properties": {
    "id": {
      "title": "Id",
      "type": "integer"
    },
    "name": {
      "default": "Jane Doe",
      "title": "Name",
      "type": "string"
    }
  },
  "required": ["id"],
  "title": "User",
  "type": "object"
}
```

This property is crucial: these JSON-formatted schemas are often passed to LLMs and the LLMs in turn use them as instructions on how to return data.

## Using annotations
As mentioned, LLMs are using JSON schemas from Pydantic as instructions on how to return data. To assist them and improve the accuracy of your returned data, it's helpful to include natural-language descriptions of objects and fields and what they're used for. Pydantic has support for this with docstrings and Fields.

We'll be using the following example Pydantic classes in all of our examples going forward:

from datetime import datetime


class LineItem(BaseModel):
    """A line item in an invoice."""

    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""

    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )

This expands to a much more complex JSON schema:

```json
{
  "$defs": {
    "LineItem": {
      "description": "A line item in an invoice.",
      "properties": {
        "item_name": {
          "description": "The name of this item",
          "title": "Item Name",
          "type": "string"
        },
        "price": {
          "description": "The price of this item",
          "title": "Price",
          "type": "number"
        }
      },
      "required": ["item_name", "price"],
      "title": "LineItem",
      "type": "object"
    }
  },
  "description": "A representation of information from an invoice.",
  "properties": {
    "invoice_id": {
      "description": "A unique identifier for this invoice, often a number",
      "title": "Invoice Id",
      "type": "string"
    },
    "date": {
      "description": "The date this invoice was created",
      "format": "date-time",
      "title": "Date",
      "type": "string"
    },
    "line_items": {
      "description": "A list of all the items in this invoice",
      "items": {
        "$ref": "#/$defs/LineItem"
      },
      "title": "Line Items",
      "type": "array"
    }
  },
  "required": ["invoice_id", "date", "line_items"],
  "title": "Invoice",
  "type": "object"
}
```

Now that you have a basic understanding of Pydantic and the schemas it generates, you can move on to using Pydantic classes for structured data extraction in LlamaIndex, starting with Structured LLMs.

# Using Structured LLMs
The highest-level way to extract structured data in LlamaIndex is to instantiate a Structured LLM. First, let's instantiate our Pydantic class as previously:

from datetime import datetime
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """A line item in an invoice."""

    item_name: str = Field(description="The name of this item")
    price: float = Field(description="The price of this item")


class Invoice(BaseModel):
    """A representation of information from an invoice."""

    invoice_id: str = Field(
        description="A unique identifier for this invoice, often a number"
    )
    date: datetime = Field(description="The date this invoice was created")
    line_items: list[LineItem] = Field(
        description="A list of all the items in this invoice"
    )

If this is your first time using LlamaIndex, let's get our dependencies:

pip install llama-index-core llama-index-llms-openai to get the LLM (we'll be using OpenAI for simplicity, but you can always use another one)
Get an OpenAI API key and set it as an environment variable called OPENAI_API_KEY
pip install llama-index-readers-file to get the PDFReader
Note: for better parsing of PDFs, we recommend LlamaParse
Now let's load in the text of an actual invoice:

from llama_index.readers.file import PDFReader
from pathlib import Path

pdf_reader = PDFReader()
documents = pdf_reader.load_data(file=Path("./uber_receipt.pdf"))
text = documents[0].text

And let's instantiate an LLM, give it our Pydantic class, and then ask it to complete using the plain text of the invoice:

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")
sllm = llm.as_structured_llm(Invoice)

response = sllm.complete(text)

response is a LlamaIndex CompletionResponse with two properties: text and raw. text contains the JSON-serialized form of the Pydantic-ingested response:

```python
json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))
```

```json
{
    "invoice_id": "Visa \u2022\u2022\u2022\u20224469",
    "date": "2024-10-10T19:49:00",
    "line_items": [
        {"item_name": "Trip fare", "price": 12.18},
        {"item_name": "Access for All Fee", "price": 0.1},
        {"item_name": "CA Driver Benefits", "price": 0.32},
        {"item_name": "Booking Fee", "price": 2.0},
        {"item_name": "San Francisco City Tax", "price": 0.21}
    ]
}
```

Note that this invoice didn't have an ID

The raw property of response (somewhat confusingly) contains the Pydantic object itself:

```python
from pprint import pprint

pprint(response.raw)
```

```python
Invoice(
    invoice_id="Visa ****4469",
    date=datetime.datetime(2024, 10, 10, 19, 49),
    line_items=[
        LineItem(item_name="Trip fare", price=12.18),
        LineItem(item_name="Access for All Fee", price=0.1),
        LineItem(item_name="CA Driver Benefits", price=0.32),
        LineItem(item_name="Booking Fee", price=2.0),
        LineItem(item_name="San Francisco City Tax", price=0.21),
    ],
)
```

Note that Pydantic is creating a full datetime object and not just translating a string.

A structured LLM works exactly like a regular LLM class: you can call chat, stream, achat, astream etc. and it will respond with Pydantic objects in all cases. You can also pass in your Structured LLM as a parameter to VectorStoreIndex.as_query_engine(llm=sllm) and it will automatically respond to your RAG queries with structured objects.

The Structured LLM takes care of all the prompting for you. If you want more control over the prompt, move on to Structured Prediction.

# Structured Prediction
Structured Prediction gives you more granular control over how your application calls the LLM and uses Pydantic. We will use the same Invoice class, load the PDF as we did in the previous example, and use OpenAI as before. Instead of creating a structured LLM, we will call structured_predict on the LLM itself; this a method of every LLM class.

Structured predict takes a Pydantic class and a Prompt Template as arguments, along with keyword arguments of any variables in the prompt template.

```python
from llama_index.core.prompts import PromptTemplate

prompt = PromptTemplate(
    "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
)

response = llm.structured_predict(
    Invoice, prompt, text=text, company_name="Uber"
)
```

As you can see, this allows us to include additional prompt direction for what the LLM should do if Pydantic isn't quite enough to parse the data correctly. The response object in this case is the Pydantic object itself. We can get the output as JSON if we want:

```python
json_output = response.model_dump_json()
print(json.dumps(json.loads(json_output), indent=2))
```

```json
{
    "invoice_id": "Uber-2024-10-10",
    "date": "2024-10-10T19:49:00",
    "line_items": [
        {"item_name": "Trip fare", "price": 12.18},
        {"item_name": "Access for All Fee", "price": 0.1}
    ]
}
```

structured_predict has several variants available for different use-cases include async (astructured_predict) and streaming (stream_structured_predict, astream_structured_predict).

## Under the hood
Depending on which LLM you use, structured_predict is using one of two different classes to handle calling the LLM and parsing the output.

FunctionCallingProgram
If the LLM you are using has a function calling API, FunctionCallingProgram will

Convert the Pydantic object into a tool
Prompts the LLM while forcing it to use this tool
Returns the Pydantic object generated
This is generally a more reliable method and will be used by preference if available. However, some LLMs are text-only and they will use the other method.

## LLMTextCompletionProgram
If the LLM is text-only, LLMTextCompletionProgram will

Output the Pydantic schema as JSON
Send the schema and the data to the LLM with prompt instructions to respond in a form the conforms to the schema
Call model_validate_json() on the Pydantic object, passing in the raw text returned from the LLM
This is notably less reliable, but supported by all text-based LLMs.

## Calling prediction classes directly
In practice structured_predict should work well for any LLM, but if you need lower-level control it is possible to call FunctionCallingProgram and LLMTextCompletionProgram directly and further customize what's happening:

textCompletion = LLMTextCompletionProgram.from_defaults(
    output_cls=Invoice,
    llm=llm,
    prompt=PromptTemplate(
        "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
    ),
)

output = textCompletion(company_name="Uber", text=text)

The above is identical to calling structured_predict on an LLM without function calling APIs and returns a Pydantic object just like structured_predict does. However, you can customize how the output is parsed by subclassing the PydanticOutputParser:

```python
from llama_index.core.output_parsers import PydanticOutputParser


class MyOutputParser(PydanticOutputParser):
    def get_pydantic_object(self, text: str):
        # do something more clever than this
        return self.output_parser.model_validate_json(text)


textCompletion = LLMTextCompletionProgram.from_defaults(
    llm=llm,
    prompt=PromptTemplate(
        "Extract an invoice from the following text. If you cannot find an invoice ID, use the company name '{company_name}' and the date as the invoice ID: {text}"
    ),
    output_parser=MyOutputParser(output_cls=Invoice),
)
```

This is useful if you are using a low-powered LLM that needs help with the parsing.

In the final section we will take a look at even lower-level calls to the extract structured data, including extracting multiple structures in the same call.

# Low-level structured data extraction
If your LLM supports tool calling and you need more direct control over how LlamaIndex extracts data, you can use chat_with_tools on an LLM directly. If your LLM does not support tool calling you can instruct your LLM directly and parse the output yourself. We'll show how to do both.

## Calling tools directly

```python
from llama_index.core.program.function_program import get_function_tool

tool = get_function_tool(Invoice)

resp = llm.chat_with_tools(
    [tool],
    # chat_history=chat_history,  # can optionally pass in chat history instead of user_msg
    user_msg="Extract an invoice from the following text: " + text,
    tool_required=True,  # can optionally force the tool call
)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_calls=False
)

outputs = []
for tool_call in tool_calls:
    if tool_call.tool_name == "Invoice":
        outputs.append(Invoice(**tool_call.tool_kwargs))

# use your outputs
print(outputs[0])
```

This is identical to structured_predict if the LLM has a tool calling API. However, if the LLM supports it you can optionally allow multiple tool calls. This has the effect of extracting multiple objects from the same input, as in this example:

```python
from llama_index.core.program.function_program import get_function_tool

tool = get_function_tool(LineItem)

resp = llm.chat_with_tools(
    [tool],
    user_msg="Extract line items from the following text: " + text,
    allow_parallel_tool_calls=True,
)

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_calls=False
)

outputs = []
for tool_call in tool_calls:
    if tool_call.tool_name == "LineItem":
        outputs.append(LineItem(**tool_call.tool_kwargs))

# use your outputs
print(outputs)
```

If extracting multiple Pydantic objects from a single LLM call is your goal, this is how to do that.

## Direct prompting
If for some reason none of LlamaIndex's attempts to make extraction easier are working for you, you can dispense with them and prompt the LLM directly and parse the output yourself, as here:

schema = Invoice.model_json_schema()
prompt = "Here is a JSON schema for an invoice: " + json.dumps(
    schema, indent=2
)
prompt += (
    """
  Extract an invoice from the following text.
  Format your output as a JSON object according to the schema above.
  Do not include any other text than the JSON object.
  Omit any markdown formatting. Do not include any preamble or explanation.
"""
    + text
)

response = llm.complete(prompt)

print(response)

invoice = Invoice.model_validate_json(response.text)

pprint(invoice)

Congratulations! You have learned everything there is to know about structured data extraction in LlamaIndex.

Other Guides
For a deeper look at structured data extraction with LlamaIndex, check out the following guides:

Structured Outputs
Pydantic Programs
Output Parsing
Bonus Track
If you're curious of learning how to boost your LLM's performance using structured inputs, check out this guide!

# Structured Input
The other side of structured data, beyond the output, is the input: many prompting guides and best practices, indeed, include some techniques such as XML tagging of the input prompt to boost the LLM's understanding of the input.

LlamaIndex offers you the possibility of natively formatting your inputs as XML snippets, leveraging banks and Jinja (make sure to have llama-index>=0.12.34 installed).

## Using Structured Input Alone
Here is a simple example of how to use structured inputs with Pydantic models:

```python
from pydantic import BaseModel
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI
from typing import Dict

template_str = "Please extract from the following XML code the contact details of the user:\n\n```xml\n{{ user | to_xml }}\n```\n\n"
prompt = RichPromptTemplate(template_str)


class User(BaseModel):
    name: str
    surname: str
    age: int
    email: str
    phone: str
    social_accounts: Dict[str, str]


user = User(
    name="John",
    surname="Doe",
    age=30,
    email="john.doe@example.com",
    phone="123-456-7890",
    social_accounts={"bluesky": "john.doe", "instagram": "johndoe1234"},
)

# check how the prompt would look like
prompt.format(user=user)

llm = OpenAI()

response = llm.chat(prompt.format_messages(user=user))

print(response.message.content)
```

As you can see, in order to employ the structured output, we need to use a Jinja expression (delimited by {{}}) with the to_xml filter (the filtering operator is |).

## Combining Structured Input with Structured Output
The combination of structured input and structured output can really boost the consistency (and thus reliability) of your LLM's output.

With this code snippet below, you can see how you can chain these two step of data structuring.

```python
from pydantic import Field
from typing import Optional


class SocialAccounts(BaseModel):
    instagram: Optional[str] = Field(default=None)
    bluesky: Optional[str] = Field(default=None)
    x: Optional[str] = Field(default=None)
    mastodon: Optional[str] = Field(default=None)


class ContactDetails(BaseModel):
    email: str
    phone: str
    social_accounts: SocialAccounts


sllm = llm.as_structured_llm(ContactDetails)

structured_response = await sllm.achat(prompt.format_messages(user=user))

print(structured_response.raw.email)
print(structured_response.raw.phone)
print(structured_response.raw.social_accounts.instagram)
print(structured_response.raw.social_accounts.bluesky)
```

If you want a more in-depth guide to structured input, check out this example notebook.