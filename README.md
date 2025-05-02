# RT DATA SCIENCE

A modern data science application with React frontend and Python FastAPI backend for movie recommendations.

## Project Structure

- `rt-data-science`: React frontend with TypeScript and Vite
- `rt-data-science-api`: Python FastAPI backend with Pinecone vector database integration

## Project Setup

### Frontend

```bash
cd rt-data-science
npm install
npm run dev
```

### Backend

```bash
cd rt-data-science-api
python -m venv .venv
.venv/scripts/activate
pip install -r requirements.txt
cp .env.example .env  # Create and configure your .env file
python main.py
```

### Environment Configuration

Create a `.env` file in the `rt-data-science-api` directory with:

```
PINECONE_API_KEY=YOUR_PINECONE_API_KEY
MOVIES_HOST=YOUR_HOST
INDEX_NAME=YOUR_INDEX_NAME
RECORD_NAMESPACE=YOUR_NAMESPACE
EMBEDDING_MODEL=llama-text-embed-v2
EMBEDDING_DIM=768
```

## Features

- Vector-based movie search and recommendations
- Dark/light theme support
- Responsive design for desktop and mobile
