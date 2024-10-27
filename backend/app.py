import os
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from PyPDF2 import PdfReader
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import json
import shutil
import traceback
from operator import itemgetter
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the uploads directory for serving PDF files
app.mount("/pdf", StaticFiles(directory=UPLOAD_DIR), name="pdf")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_async = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Elasticsearch client
# Initialize Elasticsearch client
    ###### Local Set ###############
# Initialize Elasticsearch client with basic_auth
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    basic_auth=('elastic', os.getenv('ES_PASSWORD', 'Lyx19930115'))
)

print(f"the es status is {es}")



# Define the role with full access
role_name = "aaronlu_full_access"
try:
    # First check if role exists
    existing_role = es.security.get_role(name=role_name)
    print(f"Role already exists: {existing_role}")
except Exception as e:
    # If role doesn't exist, create it
    role_body = {
        "cluster": ["all"],
        "indices": [
            {
                "names": ["*"],
                "privileges": ["all"]
            }
        ],
        "applications": [
            {
                "application": "*",
                "privileges": ["*"],
                "resources": ["*"]
            }
        ]
    }

    try:
        response = es.security.put_role(name=role_name, body=role_body)
        print(f"Role creation response: {response}")
    except Exception as e:
        print(f"Error creating role: {e}")


class ChatRequest(BaseModel):
    message: str
    theme_id: str  # Change default from "default" to "1234"
    mode: Optional[str] = "default"
    search_type: Optional[str] = "quick"
    patent_count: Optional[int] = 5

def create_es_mapping(index_name: str):
    """Create Elasticsearch index with appropriate mapping."""
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1
        },
        "mappings": {
            "properties": {
                "patent_index": {"type": "keyword"},
                "chunks": {
                    "type": "nested",
                    "properties": {
                        "chunk_index": {"type": "integer"},
                        "text": {
                            "type": "text",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            }
                        },
                        "embedding": {"type": "dense_vector", "dims": 1536},
                        "is_claims": {"type": "boolean"},
                        "is_abstract": {"type": "boolean"},
                        "is_patentability": {"type": "boolean"},
                        "is_fto": {"type": "boolean"}
                    }
                }
            }
        }
    }

    try:
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=mapping)
            print(f"Created index '{index_name}'")
        else:
            print(f"Index '{index_name}' already exists")
    except Exception as e:
        print(f"Error creating index mapping: {str(e)}")
        raise



def extract_text_from_pdf(file_path: str, filename: str) -> List[Dict]:
    """Extract text from PDF file page by page."""
    pages = []
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                text = reader.pages[page_num].extract_text()
                if text.strip():
                    pages.append({
                        "pdf_name": filename,
                        "chunk_index": page_num,
                        "text": text,
                        "is_claims": False,
                        "is_abstract": False,
                        "is_patentability": True,
                        "is_fto": False
                    })
        return pages
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

async def get_embeddings(texts: List[str]) -> List:
    """Get embeddings for a list of texts using OpenAI's API."""
    try:
        embeddings = []
        chunk_size = 8192

        for text in texts:
            if len(text) > chunk_size:
                text = text[:chunk_size]

            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")

def expand_user_question(client: OpenAI, question: str) -> List[str]:
    """Expand the user's question using GPT to generate related questions."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Generate 3 related questions that would help expand the search scope of the original question. Keep the questions concise and relevant."},
                {"role": "user", "content": question}
            ]
        )
        expanded_questions = response.choices[0].message.content.split('\n')
        return [q.strip() for q in expanded_questions if q.strip()]
    except Exception as e:
        print(f"Error expanding question: {e}")
        return [question]

def run_analysis_sync(user_message: str, results: List[Dict], search_type: str) -> str:
    """Generate analysis based on search results."""
    try:
        context = "\n\n".join([
            f"Patent {r['patent_id']}, Section {r['chunk_index']}:\n{r['text']}"
            for r in results
        ])

        system_prompts = {
            "quick": "You are a patent analyst providing quick overview analysis.",
            "patentability": "You are a patent analyst focusing on patentability analysis. Consider novelty and non-obviousness.",
            "fto": "You are a patent analyst focusing on Freedom to Operate (FTO) analysis. Consider potential infringement risks."
        }

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompts.get(search_type, system_prompts["quick"])},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}\n\nProvide a detailed analysis based on the provided context."}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analysis: {e}")
        return f"Error generating analysis: {str(e)}"
    
def verify_es_document(index_name: str, doc_id: Optional[str] = None):
    """Verify document structure in Elasticsearch"""
    try:
        if doc_id:
            doc = es.get(index=index_name, id=doc_id)
            print("\n=== Single Document Structure ===")
            print(json.dumps(doc.body, indent=2))  # Use .body to get dict
        else:
            # Get a sample document
            query = {"query": {"match_all": {}}, "size": 1}
            result = es.search(index=index_name, body=query)
            if result['hits']['hits']:
                print("\n=== Sample Document Structure ===")
                print(json.dumps(result['hits']['hits'][0], indent=2))
            else:
                print("No documents found in index")
                
        # Get mapping
        mapping = es.indices.get_mapping(index=index_name)
        print("\n=== Index Mapping ===")
        print(json.dumps(mapping.body, indent=2))  # Use .body to get dict
        
    except Exception as e:
        print(f"Error verifying document: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), theme_id: str = "default"):
    """Handle PDF upload, process it, and store in Elasticsearch."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text and create chunks
        pages = extract_text_from_pdf(file_path, file.filename)
        print("\n=== Extracted Pages ===")
        print(f"Number of pages: {len(pages)}")
        print("Sample page structure:", json.dumps(pages[0], indent=2))

        # Get embeddings for each chunk
        page_texts = [page["text"] for page in pages]
        embeddings = await get_embeddings(page_texts)
        print(f"\nGenerated {len(embeddings)} embeddings")

        # Create Elasticsearch index if it doesn't exist
        create_es_mapping(theme_id)

        # Prepare document for Elasticsearch
        doc = {
            "patent_index": file.filename,
            "chunks": [
                {
                    "chunk_index": page["chunk_index"],
                    "text": page["text"],
                    "embedding": embedding,
                    "is_claims": page.get("is_claims", False),
                    "is_abstract": page.get("is_abstract", False),
                    "is_patentability": page.get("is_patentability", True),
                    "is_fto": page.get("is_fto", False)
                }
                for page, embedding in zip(pages, embeddings)
            ]
        }

        # print("\n=== Document to be indexed ===")
        # print("Document structure:", json.dumps({
        #     "patent_index": doc["patent_index"],
        #     "chunks": [{k: v for k, v in chunk.items() if k != 'embedding'} for chunk in doc["chunks"][:1]]  # First chunk only, excluding embedding
        # }, indent=2))

        # Upload to Elasticsearch
        result = es.index(index=theme_id, document=doc)
        print(f"\nIndexing result: {result}")

        # Verify the document was indexed correctly
        verify_es_document(theme_id)

        return {
            "success": True,
            "filename": file.filename,
            "pages_processed": len(pages),
            "file_url": f"/pdf/{file.filename}"
        }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Error processing PDF: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests and perform hybrid search using Elasticsearch."""
    try:
        print("Starting chat endpoint with request:", request)  # Debug print 1
        
        # Expand the user's question
        expanded_questions = expand_user_question(client, request.message)
        user_message_expand = f"""Original question: {request.message}

        Expanded questions:
        {' '.join(f'{i+1}. {q}' for i, q in enumerate(expanded_questions))}

        Please consider all of these questions in your analysis."""

        # Get embeddings for the expanded question
        user_embedding = (await get_embeddings([user_message_expand]))[0]

                # Enforce default theme_id if not provided or if it's "default"
        # if not request.theme_id or request.theme_id == "default":
        #     request.theme_id = "1234"

        # Fix: Use assignment instead of comparison
        request.search_type = "patentability"  # Changed from == to =
        
        print(f"Search type: {request.search_type}")  # Debug print 2

        print(f"the theme_id before the patentbility search is {request.theme_id} ")

        results = patentability_search(es, request.theme_id, request.message, user_embedding, request.patent_count)
        
        print(f"Got {len(results)} results")  # Debug print 3
        
        # Print details about each result
        for idx, r in enumerate(results):
            print(f"Result {idx} fields: {list(r.keys())}")
            # print(f"Result {idx} content: {r}")  # Added full content print

        # Generate analysis
        analysis = run_analysis_sync(request.message, results, request.search_type)
        
        response_data = {
            "response": analysis,
            "sources": [
                {
                    "patent_id": r.get("patent_id", "unknown"),  # Added .get() with default
                    "chunk_index": r.get("chunk_index", 0),      # Added .get() with default
                    "score": r.get("final_score", 0.0)          # Added .get() with default
                }
                for r in results
            ]
        }
        
        print("Returning response data:", response_data)  # Debug print 4
        return response_data

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        print(f"Error type: {type(e)}")  # Added error type
        print(f"Error traceback: {traceback.format_exc()}")  # Add this import: import traceback
        raise HTTPException(status_code=500, detail=str(e))



def patentability_search(es, index_name, user_query, user_embedding, patent_count):
    logger.info(f"Starting patentability search with index: {index_name}, patent_count: {patent_count}")
    
    logger.debug("Performing BM25 search...")
    bm25_results = get_chunk_details(es, index_name, user_query, is_patentability=True, top_k=patent_count)
    logger.info(f"BM25 search returned {len(bm25_results)} results")
    
    logger.debug("Performing semantic search...")
    semantic_results = semantic_search(es, index_name, user_embedding, is_patentability=True, top_k=patent_count)
    logger.info(f"Semantic search returned {len(semantic_results)} results")
    
    if not bm25_results:
        logger.warning("BM25 search returned no results")
    if not semantic_results:
        logger.warning("Semantic search returned no results")
    
    final_results = rerank_search_results(bm25_results, semantic_results, top_k=patent_count)
    logger.info(f"Reranking completed, returned {len(final_results)} results")
    return final_results




def get_chunk_details(es, index_name, user_query, is_abstract=False, is_patentability=False, is_claims=False, top_k=None):
    logger.info(f"Starting get_chunk_details with query: {user_query[:100]}...")  # Log first 100 chars of query
    
    is_patentability = True
    must_conditions = [{"match": {"chunks.text": user_query}}]
    
    if is_abstract:
        must_conditions.extend([
            {"term": {"chunks.is_abstract": True}},
            {"term": {"chunks.is_patentability": False}},
            {"term": {"chunks.is_claims": False}}
        ])
    elif is_patentability:
        must_conditions.extend([
            {"term": {"chunks.is_abstract": False}},
            {"term": {"chunks.is_patentability": True}},
            {"term": {"chunks.is_claims": False}}
        ])
    elif is_claims:
        must_conditions.extend([
            {"term": {"chunks.is_abstract": False}},
            {"term": {"chunks.is_patentability": False}},
            {"term": {"chunks.is_claims": True}}
        ])

    initial_query = {
        "query": {
            "nested": {
                "path": "chunks",
                "query": {
                    "bool": {
                        "must": must_conditions
                    }
                },
                "inner_hits": {
                    "size": top_k
                }
            }
        },
        "_source": ["patent_index"],
        "size": top_k
    }
    
    logger.debug(f"Executing Elasticsearch query: {initial_query}")
    initial_response = es.search(index=index_name, body=initial_query)
    logger.debug(f"Got {len(initial_response['hits']['hits'])} hits from Elasticsearch")

    results = []
    for hit in initial_response['hits']['hits']:
        patent_index = hit['_source']['patent_index']
        for inner_hit in hit['inner_hits']['chunks']['hits']['hits']:
            chunk = inner_hit['_source']
            results.append({
                "patent_id": str(patent_index),
                "chunk_index": chunk['chunk_index'],
                "is_claims": chunk.get('is_claims', False),
                "is_abstract": chunk.get('is_abstract', False),
                "is_patentability": chunk.get('is_patentability', False),
                "text": chunk['text'],
                "score": inner_hit['_score']
            })

    logger.info(f"get_chunk_details returning {len(results)} results")
    return sorted(results, key=itemgetter('score'), reverse=True)


def semantic_search(es, index_name, query_embedding, is_abstract=False, is_patentability=False, is_claims=False, top_k=None):
    logger.info("Starting semantic search")
    
    is_patentability = True
    script_source = """
    cosineSimilarity(params.query_vector, 'chunks.embedding') + 1.0
    """

    must_conditions = []
    if is_abstract:
        must_conditions.extend([
            {"term": {"chunks.is_abstract": True}},
            {"term": {"chunks.is_patentability": False}},
            {"term": {"chunks.is_claims": False}}
        ])
    elif is_patentability:
        must_conditions.extend([
            {"term": {"chunks.is_abstract": False}},
            {"term": {"chunks.is_patentability": True}},
            {"term": {"chunks.is_claims": False}}
        ])
    elif is_claims:
        must_conditions.extend([
            {"term": {"chunks.is_abstract": False}},
            {"term": {"chunks.is_patentability": False}},
            {"term": {"chunks.is_claims": True}}
        ])

    query = {
        "query": {
            "nested": {
                "path": "chunks",
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "must": must_conditions
                            }
                        },
                        "script": {
                            "source": script_source,
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "inner_hits": {
                    "size": 1,
                    "_source": ["chunks.text", "chunks.is_claims", "chunks.is_abstract", "chunks.is_patentability", "chunks.chunk_index"]
                }
            }
        },
        "size": top_k,
        "_source": ["patent_index"]
    }

    #logger.debug(f"Executing semantic search query: {query}")
    
    try:
        response = es.search(index=index_name, body=query)
        logger.debug(f"Got {len(response['hits']['hits'])} hits from semantic search")
    except Exception as e:
        logger.error(f"Error during semantic search: {str(e)}")
        return []

    results = []
    for hit in response['hits']['hits']:
        patent_index = hit['_source']['patent_index']
        inner_hits = hit['inner_hits']['chunks']['hits']['hits']

        if inner_hits:
            chunk = inner_hits[0]['_source']
            result = {
                "patent_id": str(patent_index),
                "chunk_index": chunk['chunk_index'],
                "text": chunk['text'],
                "is_claims": chunk['is_claims'],
                "is_abstract": chunk.get('is_abstract', False),
                "is_patentability": chunk.get('is_patentability', False),
                "score": hit['_score']
            }
            results.append(result)
            logger.debug(f"Added result for patent {patent_index} with score {hit['_score']}")
        else:
            logger.warning(f"No inner hits for patent {patent_index}")

    logger.info(f"Semantic search returning {len(results)} results")
    return sorted(results, key=itemgetter('score'), reverse=True)




def rerank_search_results(bm25_results, semantic_results, alpha=0.7, top_k=None):
    logger.info(f"Starting reranking with {len(bm25_results)} BM25 results and {len(semantic_results)} semantic results")
    
    def normalize_scores(results):
        if not results:
            logger.warning("Attempting to normalize empty results")
            return results
            
        scores = [float(result['score']) for result in results]
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            logger.warning("Score range is 0, setting all normalized scores to 1.0")
            return [dict(result, normalized_score=1.0) for result in results]
            
        normalized = [dict(result, normalized_score=(float(result['score']) - min_score) / score_range) 
                     for result in results]
        logger.debug(f"Normalized scores range: {min([r['normalized_score'] for r in normalized])} "
                    f"to {max([r['normalized_score'] for r in normalized])}")
        return normalized

    normalized_bm25 = normalize_scores(bm25_results)
    normalized_semantic = normalize_scores(semantic_results)

    logger.debug(f"Normalized scores - BM25: {[round(r['normalized_score'], 3) for r in normalized_bm25[:3]]}...")
    logger.debug(f"Normalized scores - Semantic: {[round(r['normalized_score'], 3) for r in normalized_semantic[:3]]}...")

    combined_results = {}
    for result in normalized_bm25 + normalized_semantic:
        key = (result['patent_id'], result['chunk_index'])
        if key not in combined_results:
            combined_results[key] = result.copy()
            combined_results[key]['bm25_score'] = 0.0
            combined_results[key]['semantic_score'] = 0.0
        
        if result in normalized_bm25:
            combined_results[key]['bm25_score'] = result['normalized_score']
        if result in normalized_semantic:
            combined_results[key]['semantic_score'] = result['normalized_score']

    for key, result in combined_results.items():
        result['final_score'] = (alpha * result['bm25_score']) + ((1 - alpha) * result['semantic_score'])
        logger.debug(f"Patent {key[0]}, Chunk {key[1]}: "
                    f"BM25={result['bm25_score']:.4f}, "
                    f"Semantic={result['semantic_score']:.4f}, "
                    f"Final={result['final_score']:.4f}")

    reranked_results = sorted(combined_results.values(), key=itemgetter('final_score'), reverse=True)[:top_k]
    logger.info(f"Reranking complete. Returning {len(reranked_results)} results")
    
    if not reranked_results:
        logger.warning("No results after reranking!")
    
    return reranked_results



def get_conversation_history(theme_id: str) -> List[Dict]:
    """Get conversation history for a theme."""
    try:
        filename = f"conversation_history_{theme_id}.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return []

def save_conversation_history(theme_id: str, history: List[Dict]):
    """Save conversation history for a theme."""
    try:
        filename = f"conversation_history_{theme_id}.json"
        with open(filename, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        print(f"Error saving conversation history: {e}")

@app.get("/pdf/{filename}")
async def get_pdf(filename: str):
    """Serve PDF files."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn
    import sys
    import pathlib
    
    try:
        # Check OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not found in environment variables")
            exit(1)
            
        # Check Elasticsearch connection
        if not es.ping():
            print("Error: Could not connect to Elasticsearch")
            exit(1)
        
        print("All preliminary checks passed")
        
        # Get the current file's directory
        file_path = pathlib.Path(__file__).parent.absolute()
        sys.path.append(str(file_path))
        
        # Run with auto-reload and logging configuration
        uvicorn.run(
            "app:app", 
            host="127.0.0.1", 
            port=8000, 
            reload=True,         # Enable auto-reload
            log_level="debug",   # Set log level to debug
            workers=1,           # Use single worker
            reload_dirs=["."],   # Directories to watch for changes
            reload_delay=0.25    # Delay between reloads
        )
        
    except Exception as e:
        print(f"Startup error: {str(e)}")
        exit(1)