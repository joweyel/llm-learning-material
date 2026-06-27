# Module 2: Vector Search

## Part 1: Vector Search

### 2.1 What is Vector Search

In Module 1 we built a RAG pipeline with keyword search. Here is the full picture again:

```mermaid
flowchart LR
    U([User])
    APP[Application]
    DB[(DB)]
    DOCS[[D1 ... D5]]
    PROMPT[Build Prompt\nQuestion + Context]
    LLM[LLM]
    ANSWER([Answer])

    U -->|Question| APP
    APP -->|Query| DB
    DB -->|Retrieved Data| DOCS
    DOCS --> APP
    APP --> PROMPT
    PROMPT --> LLM
    LLM --> ANSWER
```

The search step in the middle is what this module is about. Module 1 used keyword search there. This module replaces it with vector search.

#### Keyword search and its problem

Keyword search decomposes the query into individual words and looks for documents that contain them. Ranking algorithms like BM25 or TF-IDF score documents by how many of those words appear and how often.

This works well for exact matches. It breaks down when the user phrases the question differently from how the document is written.

Example:

| Query                                                | Words                  |
| ---------------------------------------------------- | ---------------------- |
| "I just discovered the course, can I still join?"    | discover, course, join |
| "I found out about the program, can I still enroll?" | find, program, enroll  |

Both questions mean exactly the same thing. But they share almost no words. A keyword engine searching for the first query will not find documents written with the vocabulary of the second, and vice versa.

Vector search solves this.

#### What vector search does

Instead of matching words, vector search matches meaning. It converts text into a vector, a fixed-length array of numbers that captures the semantic content of the text. Texts with similar meaning produce similar vectors, regardless of which exact words were used.

The model that produces these vectors is called an **embedding model**. It is a neural network trained specifically to map meaning into a numeric space.

#### Two stages

Vector search runs in two stages:

```mermaid
flowchart LR
    subgraph OFFLINE["Offline (indexing)"]
        DOCS2["All Documents"]
        EMB1["Embedding Model"]
        IDX[("Vector Index")]
        DOCS2 --> EMB1 --> IDX
    end

    subgraph ONLINE["Online (querying)"]
        Q["User Query"]
        EMB2["Embedding Model"]
        QVEC["Query Vector"]
        SEARCH["Cosine Similarity Search"]
        RES["Top-K Documents"]
        Q --> EMB2 --> QVEC --> SEARCH
        IDX --> SEARCH
        SEARCH --> RES
    end

    classDef offline fill:#d5e8d4,stroke:#82b366,color:#2a2a2a
    classDef online  fill:#fff2cc,stroke:#d6b656,color:#2a2a2a
    classDef result  fill:#ffe6cc,stroke:#d79b00,color:#2a2a2a

    class DOCS2,EMB1,IDX offline
    class Q,EMB2,QVEC online
    class SEARCH,RES result
```

**Offline (indexing):** embed every document once and store the vectors in an index. This is done ahead of time.

**Online (querying):** embed the incoming query with the same model, then find the stored vectors closest to it by cosine similarity.

#### Cosine similarity

Cosine similarity measures the angle between two vectors in the embedding space:

- Vectors pointing in the same direction: similarity close to 1 -> similar meaning
- Vectors at right angles: similarity close to 0 -> unrelated
- Vectors pointing in opposite directions: similarity close to -1 -> opposite meaning

The higher the cosine similarity, the more semantically similar the two texts are.

#### Keyword search vs vector search

|               | Keyword search                | Vector search                           |
| ------------- | ----------------------------- | --------------------------------------- |
| Matches       | Exact words                   | Meaning                                 |
| Strengths     | Specific terms, IDs, names    | Paraphrased questions, natural language |
| Example query | "pandas dataframe"            | "How do I work with tabular data?"      |
| Index type    | Inverted index (BM25, TF-IDF) | Vector index (cosine similarity)        |
| Misses        | Synonyms and paraphrases      | Exact term matches                      |

#### When to use vector search

Vector search is usually better at capturing intent, but it adds significant operational complexity: you need an embedding model, a vector index, and more infrastructure. You will feel this throughout the module.

**Advice: do not start with vector search.** Start with keyword search. Reach for vectors only when you can show they are worth the extra cost.

In practice the two work best together. Hybrid search combines keyword and vector search and is covered in Module 6 (Best Practices).

#### What this module covers

We build vector search with three tools, in increasing operational weight:

1. **minsearch** - in-memory vector search. Simplest, good for experiments.
2. **sqlitesearch** - persistent vector search backed by SQLite. Production-friendly, same API as minsearch.
3. **PGVector** - vector search in PostgreSQL. Scalable, runs in Docker.

Then we plug vector search into the RAG pipeline from Module 1.

---

### 2.2 Embeddings

Before we can do vector search, we need to turn text into vectors. This process is called **embedding**: we embed text into a vector space. The resulting arrays of numbers are also called "embeddings."

#### The vector space idea

The idea comes from word2vec. A model learns to place words as points in a multi-dimensional space. Words with similar meanings land close to each other. Dissimilar words land far apart.

<table>
<tr>
<td align="center"><img src="images/vector_space.svg" width="280"/><br/><sub>2D projection</sub></td>
<td align="center"><img src="images/vector_space_3d.svg" width="280"/><br/><sub>3D isometric view</sub></td>
</tr>
</table>

In the diagram above, "enroll" and "join" are near each other because they mean similar things. "Docker" is far away because it is semantically unrelated. The same applies to full sentences: q1 ("I just discovered the course, can I still join?") and q2 ("I found out about the program, can I still enroll?") land close together. q3 ("How do I run Docker on Windows?") lands far away.

Documents D1-D5 from the knowledge base sit in the same space. When a user sends a query, we embed it and return the nearest documents. Those are our search results.

#### From words to sentences

An embedding model encodes the whole sentence, not individual words in isolation. This matters because the same word can mean different things in different contexts.

Example: the word "judge"

- "The judge ruled out the possibility of crime" -> legal meaning
- "LLM-as-a-judge approach to evaluate LLMs" -> ML evaluation meaning

The surrounding context shifts the embedding. The model captures this.

#### Installing sentence-transformers

We use [sentence-transformers](https://www.sbert.net/), an open-source library that runs locally with no API costs.

```bash
uv add sentence-transformers
```

This pulls in PyTorch, so the first install downloads a lot. That is expected.

#### Choosing a model

We use `all-MiniLM-L6-v2`:

- 384-dimensional vectors (compact)
- Fast on CPU
- Good quality for general English text
- Outputs normalized vectors -> dot product equals cosine similarity

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

The first run downloads the model (~80 MB) and its tokenizer from HuggingFace. After that, both load from local cache.

#### Trying it: encoding and comparing

Encode a query:

```python
q1 = "Can I still join the course after the start date?"
v1 = model.encode(q1)
# v1.shape -> (384,)
```

`v1` is an array of 384 numbers. Each number represents some concept the model learned internally. We cannot read off what any single dimension means, but two vectors with similar values point to texts about similar things.

Encode a document and compare with dot product:

```python
d  = "You don't need to register. You can just start learning and submitting homework."
dv = model.encode(d)

v1.dot(dv)   # -> 0.32  (related)
```

Now compare with an unrelated query:

```python
q2 = "How to install Docker on Windows?"
v2 = model.encode(q2)

v2.dot(dv)   # -> 0.01  (unrelated)
```

The first score (`0.32`) is higher because q1 and `d` are both about course enrollment. The second score (`0.01`) is near zero because Docker has nothing to do with registration. Near zero means the two vectors are nearly orthogonal: no shared meaning.

#### Cosine similarity and the dot product

`all-MiniLM-L6-v2` outputs **normalized** vectors (unit length). For unit vectors, dot product equals cosine similarity:

$$
\cos(\theta) = \frac{\mathbf{v}_1^T\mathbf{v}_2}{\underbrace{\|\mathbf{v}_1\|\|\mathbf{v}_2\|}_{=1}} = \mathbf{v}_1^T\mathbf{v}_2, \quad \mathbf{v}_1, \mathbf{v}_2 \in \mathbb{R}^n \\
$$


![Cosine similarity on the unit circle](./images/cosine_similarity.svg)
 
Cosine similarity values:

| Value | Angle   | Meaning                       |
| ----- | ------- | ----------------------------- |
| 1.0   | 0 deg   | same direction, similar texts |
| 0.0   | 90 deg  | perpendicular, unrelated      |
| -1.0  | 180 deg | opposite direction            |

In practice we rarely see values below 0. The embedding model maps all text into a region where vectors mostly have positive components. There is no concept of "opposite meaning" that produces a truly opposite vector.

That is the whole mechanism behind vector search: similar texts -> similar vectors -> high dot product score -> returned as top results.

---

### 2.3 Embedding Our Dataset

Now that we know how to embed individual texts, we embed the full FAQ dataset so it can be searched.

#### The dataset

The FAQ data comes from DataTalksClub courses. We load it with a helper:

```python
from ingest import load_faq_data

documents = load_faq_data()
```

`load_faq_data()` fetches the course list from the DataTalksClub website and downloads the Q&A entries for each course. Each document is a dictionary:

```python
documents[10]
# {
#   'course':   'data-engineering-zoomcamp',
#   'section':  'General Course-Related Questions',
#   'question': 'Course: How many hours per week am I expected to spend?',
#   'answer':   'It depends on your background ...',
#   'doc_id':   '316180784f'
# }
```

Total: **1350 documents** across all courses.

#### Preparing texts for embedding

An embedding model takes a single string, not a dictionary. We concatenate `question` and `answer` for each document:

```python
texts = [doc["question"] + doc["answer"] for doc in documents]
```

This gives the model both the topic (question) and the content (answer) in one shot, which tends to produce better search results than embedding either field alone.

#### Batch embedding

Passing all 1350 texts to the model at once is wasteful: no progress feedback and a large memory spike. We split into batches of 50:

```python
from tqdm.auto import tqdm

def embed_documents(texts, chunk_size=50):
    vectors = []
    for i in tqdm(range(0, len(texts), chunk_size)):
        batch = texts[i : i + chunk_size]
        vectors.extend(model.encode(batch))
    return vectors

vectors = embed_documents(texts, chunk_size=50)
```

`tqdm` wraps the loop so we see a progress bar, useful because encoding on CPU takes a few minutes for this many documents.

The result is a list of 1350 vectors, each of shape `(384,)`:

```python
len(vectors)   # 1350
vectors[0].shape  # (384,)
```

Each vector is the embedded representation of one document. In the next section we use these vectors to build a searchable index.

---

### 2.4 Vector Search

With 1350 document vectors and a query vector in hand, we need to find the most similar documents. This section builds that search step by step.

#### The naive approach

The straightforward way: loop over every document vector and compute the dot product with the query.

```python
scores_naive = [v1 @ v for v in vectors]
best_match_idx = np.argmax(scores_naive)
# Best match index: 2, similarity: 0.7629
```

This works, but it is a Python loop and slow for large datasets.

#### Matrix-vector multiplication

A much faster approach: stack all document vectors into a single NumPy matrix and do one matrix-vector multiply.

```python
X = np.array(vectors)   # shape: (1350, 384)
scores = X @ v1         # shape: (1350,)
```

`X @ v1` computes the dot product of `v1` against every row of `X` in a single call. NumPy dispatches this to an optimized BLAS routine (C/Fortran), so it is orders of magnitude faster than the Python loop while producing identical results.

#### Finding the best match

`np.argmax` returns the index of the highest score:

```python
idx = np.argmax(scores)
# idx=2, score=0.7629
documents[idx]
# {'course': 'data-engineering-zoomcamp',
#  'question': 'Course: Can I still join the course after the start date?',
#  'answer': "Yes, even if you don't register, you're still eligible ...",
#  'doc_id': '3f1424af17'}
```

The query was "Can I still join the course after the start date?". The top hit is an exact semantic match.

#### Retrieving the top-5

`np.argsort` returns indices that would sort the array in ascending order. To get the top 5, take the last 5 and reverse them:

```python
top_5_idx = np.argsort(scores)[-5:][::-1]
```

A shorter equivalent: negate the scores first, then argsort ascending and take the first 5:

```python
top_5_idx = np.argsort(-scores)[:5]
```

Results for the same query:

| Rank | Score | Course                    | Question (abbreviated)                            |
| ---- | ----- | ------------------------- | ------------------------------------------------- |
| 1    | 0.763 | data-engineering-zoomcamp | Can I still join the course after the start date? |
| 2    | 0.758 | mlops-zoomcamp            | Can I still join the course after the start date? |
| 3    | 0.719 | machine-learning-zoomcamp | The course has already started. Can I still join? |
| 4    | 0.654 | llm-zoomcamp              | I just discovered the course. Can I still join?   |
| 5    | 0.560 | data-engineering-zoomcamp | Can I follow the course after it finishes?        |

All five results are semantically about joining a course late. Ranks 1–4 are the same question rephrased across different courses. This is exactly the cross-course synonym problem that keyword search cannot handle.

Later sections add course filtering so we can restrict results to a single course before returning them.

---

### 2.5 Vector Search with `minsearch`

The NumPy approach from 2.4 works, but it is low-level: we manage the matrix manually, call `argsort`, and have no built-in way to filter by course. `minsearch` wraps all of that behind a clean API.

`minsearch` is the same library used in Module 1 for keyword search. It also ships a `VectorSearch` class that does the same brute-force cosine search we built manually, plus keyword field filtering on top.

#### Building the index

```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)
```

- `keyword_fields=["course"]` tells the index which fields can be used as exact-match filters at query time.
- `.fit(X, documents)` stores the embedding matrix `X` (shape `(1350, 384)`) alongside the original document dictionaries.

#### Searching with a filter

```python
results = vindex.search(
    v1,
    num_results=5,
    filter_dict={"course": "llm-zoomcamp"},
)
```

- `v1` is the query vector (shape `(384,)`).
- `filter_dict` pre-filters the index to only consider documents where `course == "llm-zoomcamp"` before scoring.
- `num_results=5` returns the top 5 from that filtered set.

With the filter applied, all results now belong to the LLM Zoomcamp course:

```python
# results[0]
{
  'course':   'llm-zoomcamp',
  'section':  'General Course-Related Questions',
  'question': 'I just discovered the course. Can I still join?',
  'answer':   'Yes, but if you want to receive a certificate ...',
  'doc_id':   '74eb249bbf'
}
```

Compare with the unfiltered results from 2.4: the llm-zoomcamp entry was ranked 4th there. With filtering, it rises to 1st because irrelevant courses are excluded first.

#### Why filtering matters

Without filtering, the top results are almost always the same question from different courses. A student using an LLM Zoomcamp assistant does not care about those answers. `filter_dict` fixes this with one line instead of post-processing the ranked list.

Under the hood, `VectorSearch` still does the same matrix-vector multiplication; the filter just masks rows before the dot product is computed.

---

### 2.6 RAG with Vector Search

The full RAG pipeline from Module 1 has three steps:

```
search(query) → build_prompt(query, results) → llm(prompt) → answer
```

Only the first step changes when switching to vector search. `build_prompt` and `llm` are identical. This is the payoff of keeping the pipeline modular.

#### `RAGBase` recap

`RAGBase` (in `rag_helper.py`) wires up the three steps. Its `search()` uses the minsearch keyword index with a boost dict and a `filter_dict` that restricts results to `self.course` (default: `'llm-zoomcamp'`):

```python
assistant = RAGBase(index=index, llm_client=openai_client)
assistant.rag("I just found out about the program, can I still enroll?")
# -> 'Yes, you can still enroll. If you want to receive a certificate,
#     make sure to submit your project while submissions are still open.'
```

#### Swapping in vector search

We create a subclass that overrides only `search()`. The override embeds the query string into a vector, then delegates to the `VectorSearch` index:

```python
class RAGVector(RAGBase):
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query: str, num_results: int = 5):
        query_vector = self.embedder.encode(query)
        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict={"course": self.course},
        )
```

Two differences from the base `search()`:
- the query is encoded to a vector before being passed to the index
- no `boost_dict`: there are no field weights in vector search; similarity is computed from the embedding directly

Instantiation passes the embedding model and the vector index:

```python
vector_assistant = RAGVector(
    embedder=model,      # SentenceTransformer("all-MiniLM-L6-v2")
    index=vindex,        # VectorSearch index built in 2.5
    llm_client=openai_client,
)
```

#### Running it

```python
query = "I just found out about the program, can I still enroll?"
vector_assistant.rag(query)
# -> 'Yes, but if you want to receive a certificate, you need to submit
#     your project while submissions are still open.'
```

Both keyword-RAG and vector-RAG give a correct answer here. The difference shows up for queries where the wording diverges from the documents; that is where vector search finds relevant context that keyword search misses entirely.

#### What changed

| Component      | Keyword RAG (`RAGBase`)           | Vector RAG (`RAGVector`)         |
| -------------- | --------------------------------- | -------------------------------- |
| `search()`     | minsearch text index + boost dict | embed query → VectorSearch index |
| `build_prompt` | unchanged                         | unchanged                        |
| `llm()`        | unchanged                         | unchanged                        |
| Filter         | `filter_dict={"course": ...}`     | same                             |

Because RAG is a pipeline of independent steps, swapping the retrieval backend required writing exactly one method override.

---

### 2.7 Vector Search with `sqlitesearch`

`minsearch` has three limitations that show up when moving beyond notebook experiments:

1. **Rebuilds on every startup** — embedding 1350 documents takes ~1 minute on CPU. Doing that on each application start is unacceptable.
2. **In-memory only** — everything lives in RAM. A larger dataset would not fit.
3. **Brute-force search** — for every query, the query vector is compared against every single document. At 1 000 documents this is fast, but it does not scale.

#### NN vs ANN

What we have built so far is **exact nearest-neighbor (NN) search**: compute the dot product against all documents and pick the top results. It always finds the true best matches, but it touches everything.

**Approximate nearest-neighbor (ANN) search** takes a shortcut: first identify a region of likely candidates, then score only within that region. It may occasionally miss the absolute best match, but the results are good and it is much faster at scale.

![NN (exact) vs ANN (approximate) vector search](images/nn_vs_ann.svg)

- **`Left panel`**: every document is scored against the query (expensive). 
- **`Right panel`**: a candidate region is identified first; only documents inside it are scored. The top-5 results (green) are identical, but ANN skips the grey points entirely.

`sqlitesearch` uses ANN strategies internally. It supports three modes:

| Mode   | Scale       | Algorithm                               |
| ------ | ----------- | --------------------------------------- |
| `lsh`  | up to ~100K | random hyperplane projections (default) |
| `ivf`  | 10K – 500K  | K-means clustering                      |
| `hnsw` | 10K – 1M+   | proximity graph (highest recall)        |

All modes do two-phase search: ANN candidate retrieval, then exact cosine reranking on the shortlist.

#### The ingestion / deployment split

Embedding the full dataset is done once, offline. The result is persisted to a SQLite file. The application then starts instantly by opening the existing index, without re-embedding.

```mermaid
flowchart LR
    subgraph ING["Ingestion (run once)"]
        DOCS["FAQ Documents"]
        EMB1["Embedding Model"]
        DOCS --> EMB1
    end

    DB[("faq_vectors.db")]
    EMB1 --> DB

    subgraph DEPLOY["Deployment (each query)"]
        Q["User Query"]
        EMB2["Embedding Model"]
        SEARCH["ANN Search"]
        ANS["RAG Answer"]
        Q --> EMB2 --> SEARCH --> ANS
    end

    DB --> SEARCH

    classDef ingest fill:#d5e8d4,stroke:#82b366,color:#2a2a2a
    classDef deploy fill:#fff2cc,stroke:#d6b656,color:#2a2a2a
    classDef db     fill:#dae8fc,stroke:#6c8ebf,color:#2a2a2a
    classDef result fill:#ffe6cc,stroke:#d79b00,color:#2a2a2a

    class DOCS,EMB1 ingest
    class Q,EMB2,SEARCH deploy
    class DB db
    class ANS result
```

**Ingestion** (Part 1 — run once, e.g. `vector_search.ipynb`):

```python
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db",
)
vs_index.fit(vectors, documents)  # persist to disk
vs_index.close()
```

The API is identical to `minsearch.VectorSearch` — `keyword_fields`, `.fit()`, `.search()`.

**Deployment** (Part 2 — `vector_search_persistent.ipynb`): open the existing index, no `.fit()` needed:

```python
import torch
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db",
)
```

Searching and filtering are the same as before:

```python
query_vector = model.encode("I just discovered the course. Can I still join it?")
results = vs_index.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5,
)
```

The embedding model still loads on startup — we need it to encode queries. But all 1350 document embeddings are already on disk; there is no wait.

#### RAG with sqlitesearch

`RAGVector` from section 2.6 works without any changes. Swap the index:

```python
vector_assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=openai_client,
)

vector_assistant.rag("the program has already begun, can I still sign up?")
# -> 'Yes, you can still join. You can start learning and submitting homework
#     while the form is open, even if the course has already begun ...'

vector_assistant.rag("whats queen gambit?")
# -> "I don't know."
```

The out-of-domain query correctly returns "I don't know." because nothing in the FAQ is semantically close to a chess opening.

#### minsearch vs sqlitesearch

| Feature       | `minsearch.VectorSearch`   | `sqlitesearch.VectorSearchIndex` |
| ------------- | -------------------------- | -------------------------------- |
| Storage       | In-memory (NumPy)          | On-disk (SQLite `.db` file)      |
| Search method | Exact cosine (brute force) | ANN (LSH / IVF / HNSW) + rerank  |
| Startup cost  | Re-embeds everything       | Opens existing index instantly   |
| Best for      | Notebooks, experiments     | Projects, persistent deployments |

`sqlitesearch` is a teaching library. Its only runtime dependencies are SQLite (built into Python) and NumPy, so it deploys to any host that supports SQLite at no extra cost. In production you would typically reach for PGVector or a dedicated vector database instead — which is what we cover next.

---

### 2.8 Vector Search with PGVector

Many databases support vector search: Elasticsearch has it built in, and there are dedicated stores like Qdrant and Chroma. This lesson uses **PostgreSQL** via the [pgvector](https://github.com/pgvector/pgvector) extension. Postgres is ubiquitous — most teams already run it — and pgvector gives it all the vector search capabilities of a specialized store without adding another service.

```mermaid
flowchart LR
    subgraph OFFLINE["Offline (indexing)"]
        DOCS["Documents"]
        MODEL["Embedding Model\nall-MiniLM-L6-v2"]
        DOCS --> MODEL
        MODEL -->|"vec_to_str(vector)"| PG
    end

    subgraph POSTGRES["PostgreSQL + pgvector"]
        PG[("documents\ncourse TEXT\nsection TEXT\nquestion TEXT\nanswer TEXT\nembedding vector(384)")]
        IDX["HNSW Index\nvector_cosine_ops"]
        PG --- IDX
    end

    subgraph ONLINE["Online (querying)"]
        Q["User Query"]
        QM["Embedding Model"]
        QV["Query Vector"]
        APP["Application\nRAGPgVector"]
        Q --> QM --> QV
        QV -->|"ORDER BY embedding <=> query::vector"| PG
        PG -->|"Top-K rows"| APP
    end

    classDef offlineNode fill:#d5e8d4,stroke:#82b366,color:#2a2a2a
    classDef pgNode      fill:#dae8fc,stroke:#6c8ebf,color:#2a2a2a
    classDef onlineNode  fill:#fff2cc,stroke:#d6b656,color:#2a2a2a

    class DOCS,MODEL offlineNode
    class PG,IDX pgNode
    class Q,QM,QV,APP onlineNode

    style OFFLINE  fill:#eaf4ea,stroke:#82b366,color:#2a2a2a
    style POSTGRES fill:#edf4fc,stroke:#6c8ebf,color:#2a2a2a
    style ONLINE   fill:#fffde7,stroke:#d6b656,color:#2a2a2a
```

#### Starting Postgres with pgvector

The `pgvector/pgvector:pg17` Docker image ships with the extension pre-installed:

```bash
docker run -it \
    --name pgvector \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=faq \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17
```

The `-v` flag creates a named volume so data survives container restarts.

Install the Python driver:

```bash
uv add psycopg[binary]
```

#### Connecting and activating the extension

```python
import psycopg

conn = psycopg.connect("postgresql://user:pswd@localhost:5432/faq")
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
```

`CREATE EXTENSION` turns on pgvector. It adds the `vector` column type and the similarity operators (`<=>`, `<#>`, `<->`) to the database.

#### Creating the table

```python
conn.execute("DROP TABLE IF EXISTS documents")

conn.execute("""
    CREATE TABLE documents (
        id        SERIAL PRIMARY KEY,
        course    TEXT,
        section   TEXT,
        question  TEXT,
        answer    TEXT,
        embedding vector(384)
    )
""")
```

`vector(384)` is the pgvector column type — a fixed-length array of 384 floats, matching our `all-MiniLM-L6-v2` output.

#### Inserting documents with embeddings

Postgres does not accept NumPy arrays directly. We convert each vector to a string first:

```python
def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"

for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (doc["course"], doc["section"], doc["question"], doc["answer"],
         vec_to_str(vec)),
    )

conn.commit()
```

The `::vector` cast tells Postgres to parse the string as a vector. `conn.commit()` persists the transaction.

#### Searching with cosine similarity

Encode the query and search:

```python
query = "I just discovered the course. Can I still join it?"
query_str = vec_to_str(model.encode(query))

results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, query_str)
).fetchall()
```

The `<=>` operator returns **cosine distance** (1 - cosine similarity). Ordering by ascending distance surfaces the most similar documents first. The `1 - (embedding <=> ...)` expression converts it back to a similarity score for display.

Results:

```
[llm-zoomcamp]             I just discovered the course. Can I still join?        (similarity: 0.8365)
[machine-learning-zoomcamp] The course has already started. Can I still join it?  (similarity: 0.6904)
[mlops-zoomcamp]           Course - Can I still join the course after the start?  (similarity: 0.6043)
[data-engineering-zoomcamp] Course: Can I still join the course after the start?  (similarity: 0.5959)
```

#### Filtering by course

Filtering is a plain SQL `WHERE` clause:

```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE course = %s
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, "llm-zoomcamp", query_str)
).fetchall()
```

Three parameters here: query vector appears twice (for the SELECT expression and for ORDER BY), course appears once.

#### HNSW index for faster search

The queries above run brute-force over all rows. For larger datasets, create an HNSW index:

```python
conn.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
""")
```

This is the same HNSW algorithm from section 2.7. After the index is built, Postgres switches from brute-force to ANN automatically. No changes to the query are needed.

#### Wrapping search in a function

```python
def pgvector_search(query, course="llm-zoomcamp", num_results=5):
    query_vector = model.encode(query)
    query_str = vec_to_str(query_vector)
    rows = conn.execute(
        """
        SELECT course, section, question, answer
        FROM documents
        WHERE course = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (course, query_str, num_results)
    ).fetchall()
    return [
        {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
        for r in rows
    ]
```

#### RAG with PGVector

`RAGPgVector` overrides `search()` to use the Postgres connection instead of an index object. We set `index=None` because `RAGBase.__init__` requires the argument but we never use it:

```python
from rag_helper import RAGBase

class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)
        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()
        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]
```

```python
vector_assistant = RAGPgVector(
    embedder=model,
    conn=conn,
    llm_client=openai_client,
)

vector_assistant.rag("the program has already begun, can I still sign up?")
# -> 'Yes, you can still join even if the program has already begun.
#     If you want a certificate, make sure to submit your project while
#     submissions are still open.'
```

#### All three backends compared

| Feature       | `minsearch`         | `sqlitesearch`              | PGVector                            |
| ------------- | ------------------- | --------------------------- | ----------------------------------- |
| Storage       | In-memory           | SQLite file                 | PostgreSQL database                 |
| Setup         | None                | None                        | Docker + psycopg                    |
| Search method | Exact (brute force) | ANN (LSH/IVF/HNSW) + rerank | Brute force; HNSW index optional    |
| Persistence   | No                  | Yes                         | Yes                                 |
| Filtering     | `filter_dict`       | `filter_dict`               | SQL `WHERE`                         |
| Concurrency   | No                  | Limited                     | Yes (full ACID)                     |
| Best for      | Notebooks           | Pet projects, free tiers    | Production, existing Postgres infra |

---

### 2.9 ONNX Embedder (Optional)

`sentence-transformers` pulls in PyTorch and a pile of NVIDIA libraries. ONNX Runtime serves the same model without that weight:

| Setup                 | Size   | Packages |
| --------------------- | ------ | -------- |
| sentence-transformers | 4.8 GB | 58       |
| ONNX Runtime          | 147 MB | 27       |

33x smaller, identical embeddings. Worth it for production deployments; for notebooks, sentence-transformers is fine.

The ONNX workflow lives in [`code/llm-zoomcamp-onnx/`](./code/llm-zoomcamp-onnx/) and the notebook [`vector_search_onnx.ipynb`](./code/llm-zoomcamp-onnx/vector_search_onnx.ipynb) walks through the full pipeline.

**How `Embedder` works internally:**

1. `Tokenize`: text to integer IDs + attention mask
2. `ONNX inference`: run the model graph on CPU
3. `Mean pooling`: average token embeddings weighted by attention mask
4. `L2 normalize`: divide by L2 norm so dot product equals cosine similarity

The `encode` / `encode_batch` interface is identical to `SentenceTransformer.encode()`, so the rest of the pipeline is unchanged.

---

### 2.10 Next Steps

**What we built in this module:**

- Learned what embeddings are and how they turn text into vectors
- Generated embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)
- Built vector search with NumPy, `minsearch`, `sqlitesearch`, and PGVector
- Integrated vector search into the RAG pipeline via `RAGVector` and `RAGPgVector`

#### When to use vector search

Text search is simple, fast, and sufficient for many applications. Start there.

Vector search adds real overhead: you need an embedding model, you need to compute and store embeddings, and at query time you encode the query before you can search. Don't take it on without a reason.

A reasonable progression:

| Version | Approach | When to move on |
| --- | --- | --- |
| v1 | Text search (keyword) | When users phrase queries differently from documents |
| v2 | Vector search (semantic) | When evaluation shows a meaningful improvement |
| v3 | Hybrid search (text + vector) | Typically outperforms either alone; covered in Module 6 |

The right time to move from one version to the next is when evaluation shows it is justified. A later module covers how to measure retrieval quality. With numbers in hand you can tell a marginal gain from a real one.

#### Hybrid search and reranking

Once you have both, you can combine them. **Hybrid search** (e.g. Reciprocal Rank Fusion) merges results from both methods. **Reranking** adds a second model that re-scores the retrieved candidates for relevance. Both are covered in Module 6 (Best Practices).

#### What to try next

- Compare text search and vector search on your own data
- Experiment with different embedding models (see the table in section 2.9)
- Try PGVector with a larger dataset and benchmark the HNSW index
- Explore other vector databases: Elasticsearch, Qdrant, Weaviate, Chroma. The concepts are the same: embed documents, store vectors, search by similarity

