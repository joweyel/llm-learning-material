## Part 1: RAG

### 1.1 Introductions

#### What are LLMs

An LLM (Large Language Model) is a neural network trained on vast amounts of text data. Its core task is next-token prediction: given some input text, it predicts the most likely continuation.

A simple analogy: a search engine's autocomplete suggests the next word based on what millions of users typed before. An LLM does the same but at a much larger scale with billions of parameters trained on most of the internet, making it seem like you are talking to something that truly understands you.

In this course, LLMs are treated as black boxes: text in, text out. We call them via API and don't look inside.

**Limitations:**

- **Knowledge cutoff:** only knows what was in its training data, ignorant of anything more recent
- **No access to private data:** cannot see your documents or databases unless you explicitly provide them
- **Hallucinations:** can confidently produce wrong answers

#### The Project

RAG (Retrieval-Augmented Generation) addresses the limitations above by fetching relevant documents at query time and handing them to the LLM as context. Instead of hoping the model memorized the answer, we retrieve the right information and let it generate a grounded response.

The concrete project in this module is a FAQ agent for the course itself. A student asks something like "when does the course start?" and the agent answers from prepared FAQ data.

The module is split into two parts:

- **Part 1:** build a RAG pipeline with keyword search from scratch
- **Part 2:** make it agentic so the LLM decides when and what to search, instead of running the same fixed flow every time

### 1.2 Environment Setup

Install `uv` (fast Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create project and add dependencies:

```bash
uv init
uv add requests minsearch openai jupyter python-dotenv
```

Store the API key in `.env` (never commit this file):

```
OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

Add `.env` to `.gitignore`. Start Jupyter with:

```bash
uv run jupyter notebook
```

Verify the setup:

```python
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
openai_client = OpenAI()
```

For alternative providers (e.g. Groq), set the key and override `base_url`:

```python
openai_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
```

### 1.3 What is RAG

A plain LLM only knows what was in its training data. Ask it something specific to your own data (e.g. course enrollment policies) and it either guesses or refuses. The fix: give it the relevant information as part of the prompt.

```mermaid
flowchart LR
    U([User])

    APP[Application]

    DB[(DB)]
    DOCS[[D1 ... D5]]

    PROMPT[Build Prompt<br/>Question + Context]

    LLM[LLM]

    ANSWER([Answer])

    U -->|Question| APP

    APP -->|Query| DB
    DB -->|Retrieved Data| DOCS
    DOCS --> APP

    APP --> PROMPT
    PROMPT --> LLM

    LLM --> ANSWER
    ANSWER --> U
```

**RAG (Retrieval-Augmented Generation)** automates exactly that:

1. Take the user's question
2. Search a knowledge base for relevant documents
3. Inject those documents into the prompt as context
4. Let the LLM generate a grounded answer

In code, the full pipeline collapses to three steps:

```python
def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
```

The three components are independent and swappable: any search engine, any prompt template, any LLM. Answer quality is directly tied to retrieval quality: if the wrong documents are retrieved, the LLM gets the wrong context and the answer will be wrong.

### 1.4 The Course FAQ Dataset

The FAQ originated from students repeatedly asking the same questions in Slack. Those were collected into a single document so people can search before asking. Older courses like Data Engineering Zoomcamp have run for five cohorts and accumulated hundreds of entries, making manual search tedious. That is exactly the problem the RAG system solves.

The data is available as JSON from the DataTalks.Club website:

```python
import requests

docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()

documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"""{url_prefix}{course["path"]}"""
    course_response = requests.get(course_url)
    course_response.raise_for_status()
    documents.extend(course_response.json())
```

`raise_for_status()` raises an error immediately if a request fails instead of silently continuing.

Each document has these fields:

| Field      | Description                                    |
| ---------- | ---------------------------------------------- |
| `id`       | Unique identifier                              |
| `course`   | Course slug (e.g. `machine-learning-zoomcamp`) |
| `section`  | Section of the course                          |
| `question` | The FAQ question                               |
| `answer`   | The FAQ answer                                 |

The `course` field enables filtering by course, `section` adds useful ranking context. In the RAG pipeline: index all documents, search on the user's question, pass the top results as context to the LLM.

> **Data preparation note:** This dataset is already clean because the instructor maintains the website. In real projects, expect to spend a lot of time scraping, parsing PDFs, and chunking documents before getting to the GenAI part. The course skips this step on purpose to stay focused on the RAG pipeline itself.

### 1.5 Search

#### How Search Works

Every search engine does the same thing: score every document against the query and return the top N results.

```python
score = sim(query, document)
```

What differs between engines is what `sim` computes:

- **Text/lexical search:** counts shared words (exact surface match)
- **Vector/semantic search:** compares meaning (covered in Module 2)

Limitation of text search: "Can I still join?" and "Is it possible to enroll late?" mean the same thing but share almost no keywords, so a text search engine would struggle to connect them.

#### Why Not Send All Documents to the LLM?

With ~1100 documents, sending everything to the LLM would be expensive, slow, and confusing for the model. Search finds the most relevant candidates first.

#### minsearch

[minsearch](https://github.com/alexeygrigorev/minsearch) is a lightweight in-memory search engine built for teaching. It uses the same concepts as Elasticsearch (text fields, keyword fields, boosting, filtering) but runs anywhere Python runs, without Docker. Useful for small datasets.

**Text fields** are tokenized and ranked by relevance. **Keyword fields** require an exact match, like a SQL `WHERE` clause, and restrict the search space to a subset (e.g. only results from one course).

```python
from minsearch import Index

index = Index(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"]
)

index.fit(documents)
```

`fit` comes from scikit-learn terminology: fitting the index on the documents.

#### Searching and Boosting

```python
search_results = index.search(
    question,
    boost_dict={"question": 2.0, "section": 0.5},
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

- `boost_dict`: relative field importance. Default is 1.0. A match in `question` (boost 2.0) counts twice as much as in `answer`. A match in `section` (boost 0.5) counts half as much.
- `filter_dict`: hard filter on keyword fields. Only results from the specified course are returned.

#### Wrapping it as a Function

```python
def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5
    )
```

This is the first component of the RAG pipeline.

### 1.6 Building a Prompt

The prompt is split into two parts:

- **Instructions (system prompt):** fixed for every request, tells the LLM how to behave
- **User prompt:** changes with every request, carries the question and retrieved context

Keeping them separate makes the fixed part easy to reuse and the changing part easy to rebuild each time.

```python
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""
```

The `context` is built by converting the list of search result dicts into a readable string:

```python
def build_context(search_results):
    lines = []
    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")
    return "\n".join(lines).strip()
```

The full `build_prompt` function combines both:

```python
def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
    return prompt.strip()
```

`.strip()` removes leading/trailing whitespace from the template literals.

The prompt is the bridge between search and the LLM. A bad prompt lets the LLM ignore the context and hallucinate. Prompt engineering is iterative: later in the course, evaluation metrics replace guesswork.

## Part 2: Agents

## Homework

## Optional
