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
from typing import List, Tuple, Dict, Optional
from openai import AsyncOpenAI
import asyncio
from collections import defaultdict


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

    
def verify_es_document(index_name: str, doc_id: Optional[str] = None):
    """Verify document structure in Elasticsearch"""
    try:
        if doc_id:
            doc = es.get(index=index_name, id=doc_id)
            #print("\n=== Single Document Structure ===")
            #print(json.dumps(doc.body, indent=2))  # Use .body to get dict
        else:
            # Get a sample document
            query = {"query": {"match_all": {}}, "size": 1}
            result = es.search(index=index_name, body=query)
            #if result['hits']['hits']:
                #print("\n=== Sample Document Structure ===")
                #print(json.dumps(result['hits']['hits'][0], indent=2))
            #else:
            #    print("No documents found in index")
                
        # Get mapping
        mapping = es.indices.get_mapping(index=index_name)
        #print("\n=== Index Mapping ===")
        #print(json.dumps(mapping.body, indent=2))  # Use .body to get dict
        
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
        #print(f"\nIndexing result: {result}")

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
    

############################ Chat Response ##################################


# Define proper Pydantic models with explicit types
class ChatRequest(BaseModel):
    message: str
    theme_id: str  # Change default from "default" to "1234"
    mode: Optional[str] = "default"
    search_type: Optional[str] = "quick"
    patent_count: Optional[int] = 5


class Source(BaseModel):
    patent_id: str
    chunk_index: int
    final_score: float
    content: str = ""

class ChatResponse(BaseModel):
    response: str
    sources: List[Source]

def get_system_message(search_type: str) -> str:
    """
    Get system message based on search type with proper error handling.
    """
    try:
        base_messages = {
            "patentability": """You are an AI assistant specializing in patentability analysis. 
            Focus on identifying relevant prior art and analyzing its implications for patentability.""",
            
            "infringement": """You are an AI assistant specializing in patent infringement analysis. 
            Focus on identifying potential infringement issues and analyzing claim coverage.""",
            
            "validity": """You are an AI assistant specializing in patent validity analysis. 
            Focus on analyzing the validity of patent claims and identifying potential invalidity arguments.""",
            
            "general": """You are an AI assistant specializing in patent analysis. 
            Provide comprehensive analysis of patent content and its implications."""
        }
        
        return base_messages.get(search_type.lower(), base_messages["general"])
    
    except Exception as e:
        logger.error(f"Error in get_system_message: {str(e)}")
        return """You are an AI assistant specializing in patent analysis. 
        Provide detailed, accurate analysis of patent content while maintaining proper citations and references."""
    


def create_single_patent_prompt(user_query: str, patent: Dict, search_type: str) -> str:
    """
    Creates a prompt with proper string handling and initialization.
    """
    # Initialize base prompt with f-string
    base_prompt = f"""
Analyze the following patent in relation to this query: "{user_query}"

Patent ID: {patent['patent_id']}
Available Pages: {patent['chunk_indices']}

Content by page:"""
    
    # Build page-specific content sections using list comprehension and join
    page_contents = []
    for chunk in patent['chunks']:
        chunk_text = chunk.get('text', '').strip()
        if chunk_text:  # Only add non-empty chunks
            page_contents.append(f"\nPAGE {chunk.get('chunk_index')}:\n{chunk_text}")
    
    # Join all content parts
    full_prompt = base_prompt + ''.join(page_contents)
    
    # Add analysis instructions
    analysis_instructions = """

Please provide a detailed analysis that:
1. References specific pages when discussing content
2. Uses direct quotes with page numbers to support key points
3. Makes connections between content from different pages when relevant
4. Uses the format "On page X: [finding/content]" when referencing specific sections

Focus on how this patent relates to the query, highlighting the most relevant sections with page references.
"""
    
    return full_prompt + analysis_instructions
    

async def run_analysis_sync(message: str, results: List[dict], search_type: str) -> List[Tuple[str, List[dict]]]:
    """
    Generate analysis with references for each section using OpenAI.
    First combines chunks from the same patent before analysis.
    """
    try:
        # Log initial results details
        unique_patents = len({r['patent_id'] for r in results[:5]})
        unique_chunks = len([(r['patent_id'], r['chunk_index']) for r in results[:5]])
        logger.info(f"Initial results stats:")
        logger.info(f"Total results in top 5: {len(results[:5])}")
        logger.info(f"Unique patent IDs: {unique_patents}")
        logger.info(f"Unique patent chunks: {unique_chunks}")
        
        # Print detailed breakdown
        patent_chunk_mapping = defaultdict(list)
        for r in results[:5]:
            patent_chunk_mapping[r['patent_id']].append(r['chunk_index'])
        
        logger.info("Detailed patent-chunk breakdown:")
        for patent_id, chunks in patent_chunk_mapping.items():
            logger.info(f"Patent {patent_id}: {len(chunks)} chunks - Indices: {chunks}")

        if not results:
            return [("No relevant documents were found for your query.", [])]

        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Preprocess and combine chunks from same patent
        combined_results = combine_patent_chunks(results[:5])
        
        # Log combined results details
        logger.info(f"\nAfter combining chunks:")
        logger.info(f"Number of unique patents for analysis: {len(combined_results)}")
        for patent in combined_results:
            logger.info(f"Patent {patent['patent_id']}: Combined {len(patent['chunks'])} chunks - "
                       f"Indices: {patent['chunk_indices']} - Score: {patent['final_score']:.3f}")
        
        # Analyze patents
        analyses = await asyncio.gather(*[
            analyze_single_patent_async(message, patent_data, search_type, client)
            for patent_data in combined_results
        ])
        
        # Format the analyses
        formatted_analyses = []
        for analysis, patent_data in zip(analyses, combined_results):
            if analysis:  # Only include successful analyses
                formatted_analyses.append((analysis, patent_data['chunks']))
        
        # Add overall conclusion
        if formatted_analyses:
            conclusion = "\nOverall Conclusion:\n"
            conclusion += analyze_conclusion(message, results[:5])
            formatted_analyses.append((conclusion, results[:5]))
        
        logger.info(f"\nAnalysis complete. Generated {len(formatted_analyses)} sections")
        return formatted_analyses

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return [("An error occurred during analysis.", [])]
    

def combine_patent_chunks(results: List[dict]) -> List[dict]:
    """
    Combines chunks with proper string handling and initialization.
    """
    # Group chunks by patent_id
    patent_groups = defaultdict(list)
    for result in results:
        if result.get('patent_id'):  # Only process results with valid patent_id
            patent_groups[result['patent_id']].append(result)
    
    combined_results = []
    for patent_id, chunks in patent_groups.items():
        try:
            # Sort chunks by chunk_index (page number)
            sorted_chunks = sorted(chunks, key=lambda x: x.get('chunk_index', 0))
            
            # Log combining process
            logger.info(f"\nCombining pages for patent {patent_id}:")
            logger.info(f"Found {len(chunks)} pages with numbers: {[chunk.get('chunk_index') for chunk in sorted_chunks]}")
            
            # Build combined text with proper string handling
            combined_text_parts = []
            for chunk in sorted_chunks:
                chunk_text = chunk.get('text', '').strip()
                if chunk_text:  # Only add non-empty chunks
                    combined_text_parts.append(f"PAGE {chunk.get('chunk_index')}:\n{chunk_text}")
            
            # Create combined patent entry
            combined_patent = {
                'patent_id': patent_id,
                'text': '\n\n'.join(combined_text_parts),
                'final_score': max((chunk.get('final_score', 0) for chunk in sorted_chunks), default=0),
                'chunks': sorted_chunks,
                'chunk_indices': [chunk.get('chunk_index') for chunk in sorted_chunks]
            }
            
            logger.info(f"Combined {len(chunks)} pages of content")
            logger.info(f"Using highest score: {combined_patent['final_score']:.3f}")
            
            combined_results.append(combined_patent)
            
        except Exception as e:
            logger.error(f"Error combining chunks for patent {patent_id}: {str(e)}")
            logger.error("Error details:", exc_info=True)
            continue
    
    # Sort combined results by max score
    combined_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    return combined_results

async def analyze_single_patent_async(user_query: str, patent: Dict, search_type: str, client: AsyncOpenAI) -> str:
    """
    Enhanced analysis function with Markdown formatting for better frontend display.
    """
    try:
        logger.info(f"\nAnalyzing patent {patent['patent_id']}:")
        logger.info(f"Analyzing content from pages: {patent['chunk_indices']}")
        
        # Validate patent content
        if not patent.get('text', '').strip():
            logger.warning(f"Empty or invalid patent content for {patent.get('patent_id', 'unknown')}")
            return f"Unable to analyze Patent {patent.get('patent_id', 'unknown')} - Invalid content"

        # Create page reference section with Markdown formatting
        page_references = []
        for chunk in patent['chunks']:
            chunk_text = chunk.get('text', '').strip()
            if chunk_text:
                preview_text = chunk_text[:50]
                if len(chunk_text) > 50:
                    last_space = preview_text.rfind(' ')
                    if last_space > 0:
                        preview_text = preview_text[:last_space]
                    preview_text += "..."
                
                page_references.append(f"* **Page {chunk.get('chunk_index', 'unknown')}**:\n  > {preview_text}")

        # Format page references with proper Markdown
        page_references_text = "\n".join(page_references)

        # Get base system message with fallback
        base_system_message = get_system_message(search_type)
        if base_system_message is None:
            base_system_message = """You are an AI assistant specializing in patent analysis. 
            Provide detailed, accurate analysis of patent content while maintaining proper citations and references."""

        # Create complete system message
        system_message = f"""{base_system_message}

IMPORTANT: Your analysis must:
1. Reference specific page numbers when quoting or referring to content
2. Use the format 'On page X: [content/finding]' when discussing specific sections
3. Provide direct quotes with page references to support key points
4. Make clear connections between different pages when relevant
5. Focus on the most relevant sections that address the query
6. Use markdown formatting for emphasis and structure when appropriate
"""

        prompt = create_single_patent_prompt(user_query, patent, search_type)
        
        logger.info(f"Sending request to OpenAI for patent {patent['patent_id']}")
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"Successfully received analysis for patent {patent['patent_id']}")

        # Create formatted response with Markdown
        formatted_response = f"""
# Patent Analysis: {patent['patent_id']} ({search_type.upper()})

## 📄 Referenced Pages
{page_references_text}

---

## 📋 Detailed Analysis
{analysis}

---
"""

        return formatted_response.strip()
    
    except Exception as e:
        logger.error(f"Error analyzing patent: {str(e)}")
        logger.error("Error details:", exc_info=True)
        return f"Error analyzing patent {patent.get('patent_id', 'unknown')}: {str(e)}"
    
    

def analyze_conclusion(message: str, refs: List[dict]) -> str:
    """
    Generate a conclusion based on the analysis.
    """
    if not refs:
        return "No relevant prior art was found for this invention."
    
    # Sort by score and take top references
    top_refs = sorted(refs, key=lambda x: x.get('final_score', 0), reverse=True)[:3]
    conclusion = "Based on the analysis of the most relevant prior art:\n"
    
    for ref in top_refs:
        conclusion += (f"• Document {ref.get('patent_id', 'unknown')}: "
                      f"Relevance score {ref.get('final_score', 0):.2f}\n")
    
    return conclusion


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info("Starting chat endpoint with request: %s", request)
        
        expanded_questions = expand_user_question(client, request.message)
        user_message_expand = f"""Original question: {request.message}
        Expanded questions:
        {' '.join(f'{i+1}. {q}' for i, q in enumerate(expanded_questions))}
        Please consider all of these questions in your analysis."""
        
        user_embedding = (await get_embeddings([user_message_expand]))[0]
        request.search_type = "patentability"
        
        results = patentability_search(es, request.theme_id, request.message, 
                                    user_embedding, request.patent_count)
        
        logger.info(f"Got {len(results)} results")
        
        if not results:
            logger.warning("No results found from search")
            return ChatResponse(
                response="I couldn't find any relevant information to answer your question.",
                sources=[]
            )
        
        # Generate analysis with inline references
        analysis_sections = await run_analysis_sync(request.message, results, request.search_type)
        
        formatted_response = ""
        all_sources: List[Source] = []
        
        for section_text, section_refs in analysis_sections:
            formatted_response += section_text + "\n\n"
            
            for ref in section_refs:
                source = Source(
                    patent_id=ref.get('patent_id', 'unknown'),
                    chunk_index=ref.get('chunk_index', 0),
                    final_score=float(ref.get('final_score', 0.0)),
                    content=ref.get('content', '')
                )
                if not any(s.patent_id == source.patent_id and 
                         s.chunk_index == source.chunk_index for s in all_sources):
                    all_sources.append(source)
        
        response_data = ChatResponse(
            response=formatted_response.strip(),
            sources=all_sources
        )
        
        return response_data
        
    except Exception as e:
        logger.error("Error in chat endpoint: %s", str(e))
        logger.error("Error type: %s", type(e))
        logger.error("Traceback: %s", traceback.format_exc())
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
    
    # logger.debug(f"Executing Elasticsearch query: {initial_query}")
    initial_response = es.search(index=index_name, body=initial_query)
    # logger.debug(f"Got {len(initial_response['hits']['hits'])} hits from Elasticsearch")

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
    """
    Rerank search results combining BM25 and semantic search scores.
    """
    logger.info(f"Starting reranking with {len(bm25_results)} BM25 results and {len(semantic_results)} semantic results")
    
    def normalize_scores(results):
        if not results:
            logger.warning("Attempting to normalize empty results")
            return []  # Return empty list instead of the input
        
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
    
    # Handle case when only semantic results exist
    if not bm25_results and semantic_results:
        logger.info("Only semantic results available, using semantic scores directly")
        normalized_semantic = normalize_scores(semantic_results)
        for result in normalized_semantic:
            result['final_score'] = result['normalized_score']  # Use semantic score as final score
        return normalized_semantic if not top_k else normalized_semantic[:top_k]
    
    normalized_bm25 = normalize_scores(bm25_results)

    normalized_semantic = normalize_scores(semantic_results)
    
    combined_results = {}

    for result in normalized_bm25 + normalized_semantic:
        key = (result['patent_id'], result['chunk_index'])
        if key not in combined_results:
            combined_results[key] = result.copy()
            combined_results[key]['bm25_score'] = 0.0
            combined_results[key]['semantic_score'] = 0.0
            combined_results[key]['final_score'] = 0.0  # Initialize final_score
        
        if result in normalized_bm25:
            combined_results[key]['bm25_score'] = result['normalized_score']
        if result in normalized_semantic:
            combined_results[key]['semantic_score'] = result['normalized_score']
            # When only semantic results exist, use semantic score directly
            if not normalized_bm25:
                combined_results[key]['final_score'] = result['normalized_score']

    if normalized_bm25 and normalized_semantic:
        for result in combined_results.values():
            result['final_score'] = (alpha * result['bm25_score']) + ((1 - alpha) * result['semantic_score'])

    reranked_results = sorted(combined_results.values(), key=lambda x: x['final_score'], reverse=True)
    if top_k:
        reranked_results = reranked_results[:top_k]

    logger.info(f"Reranking complete. Returning {len(reranked_results)} results")
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