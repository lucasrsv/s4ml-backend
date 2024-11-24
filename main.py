from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import boto3
import io
import os
import uvicorn

# FastAPI app
app = FastAPI()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY_ID')
AWS_BUCKET_NAME = 'se4ml-data'
AWS_REGION = 'us-east-2'
S3_EMBEDS_KEY = 'backend/compressed_array.npz'
S3_DF_KEY = 'backend/compressed_dataframe.csv.gz'

def download_file_from_s3(bucket_name, s3_key):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    
    file_obj = io.BytesIO()
    s3_client.download_fileobj(bucket_name, s3_key, file_obj)
    file_obj.seek(0)
    return file_obj

# Paths to pre-saved data
PATH_TO_EMBEDS = 'data/compressed_array.npz'
PATH_TO_DF = 'data/compressed_dataframe.csv.gz'

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Load embeddings and DataFrame
def download_file_from_s3(bucket_name, s3_key):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    
    file_obj = io.BytesIO()
    s3_client.download_fileobj(bucket_name, s3_key, file_obj)
    file_obj.seek(0)
    return file_obj
        
embeddings_file = download_file_from_s3(AWS_BUCKET_NAME, S3_EMBEDS_KEY)
embeddings = np.load(embeddings_file)['array_data']

df_file = download_file_from_s3(AWS_BUCKET_NAME, S3_DF_KEY)
df_data = pd.read_csv(df_file, compression='gzip')

# Initialize FAISS index
embed_length = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(embed_length)
faiss_index.add(embeddings)

# CORS Configuration
origins = [
    "https://se4ml-frontend-aea8e8a17cba.herokuapp.com",
    "http://localhost",  # Local development
    "http://localhost:3000",  # React app or similar running on port 3000
    "https://example.com",  # Add any other domains that should be allowed
]

# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins to allow
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Request schema
class SearchRequest(BaseModel):
    query_text: str
    num_results_to_print: int = 10
    top_k: int = 10

# Response schema
class SearchResult(BaseModel):
    arxiv_id: str
    link_to_pdf: str
    cat_text: str
    title: str
    abstract: str

# Topic schema
class Topic(BaseModel):
    topic: str
    score: float

def run_faiss_search(query_text, top_k):
    query = [query_text]
    query_embedding = model.encode(query)
    scores, index_vals = faiss_index.search(query_embedding, top_k)
    return index_vals[0]  # Return the list of top_k indices


def run_rerank(index_vals_list, query_text):
    chunk_list = list(df_data['prepared_text'])
    pred_strings_list = [chunk_list[item] for item in index_vals_list]

    cross_input_list = [[query_text, item] for item in pred_strings_list]
    df = pd.DataFrame(cross_input_list, columns=['query_text', 'pred_text'])
    df['original_index'] = index_vals_list

    cross_scores = cross_encoder.predict(cross_input_list)
    df['cross_scores'] = cross_scores

    df_sorted = df.sort_values(by='cross_scores', ascending=False).reset_index(drop=True)

    pred_list = []
    for i in range(len(df_sorted)):
        original_index = df_sorted.loc[i, 'original_index']
        arxiv_id = str(df_data.loc[original_index, 'id'])  # Ensure arxiv_id is a string
        pred_list.append({
            "arxiv_id": arxiv_id,
            "link_to_pdf": f"https://arxiv.org/pdf/{arxiv_id}",
            "cat_text": df_data.loc[original_index, 'cat_text'],
            "title": df_data.loc[original_index, 'title'],
            "abstract": df_sorted.loc[i, 'pred_text']
        })
    return pred_list


@app.post("/search", response_model=List[SearchResult])
def search_arxiv(request: SearchRequest):
    try:
        index_vals_list = run_faiss_search(request.query_text, request.top_k)
        pred_list = run_rerank(index_vals_list, request.query_text)
        return pred_list[:request.num_results_to_print]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))