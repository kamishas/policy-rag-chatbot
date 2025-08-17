# Air Force RAG Assistant - Frontend

A modern React-based chat interface for the Air Force RAG (Retrieval-Augmented Generation) Assistant that connects to the FastAPI backend for real-time AI responses.

## Features

- 🎨 **Modern Glass-morphism UI** - Beautiful, professional interface with gradient backgrounds
- 💬 **Real-time Chat Interface** - Live connection to FastAPI backend with LLM responses
- 📚 **Source Citations** - Displays document sources with similarity scores and page references
- ⚡ **Performance Metrics** - Shows processing time for each AI response
- 🔄 **Error Handling** - Graceful error states when backend is unavailable
- 📱 **Responsive Design** - Works perfectly on desktop and mobile devices
- 🎯 **Suggestion Chips** - Quick-start questions for common Air Force queries
- ⚡ **Fast Performance** - Built with Vite for lightning-fast development and builds
- 🎨 **Tailwind CSS** - Utility-first CSS framework for rapid styling

## Tech Stack

- **React 19** - Latest React with modern hooks and features
- **Vite** - Lightning-fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Fetch API** - For backend communication
- **ESLint** - Code quality and consistency

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx          # Main chat interface with backend integration
│   ├── main.jsx         # React app entry point
│   └── index.css        # Global styles and Tailwind directives
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
└── README.md           # This file
```

## Backend Integration

### API Connection
The frontend connects to the FastAPI backend running on `http://localhost:8000`:

- **Endpoint**: `POST /api/chat`
- **Request**: `{ "message": "user question" }`
- **Response**: 
```json
{
  "response": "AI-generated answer",
  "sources": [
    {
      "document": "document-url",
      "doc_type": "AFI",
      "similarity_score": 0.85,
      "page": "Chapter 2",
      "chunk_id": 1
    }
  ],
  "processing_time": 1.23
}
```

### Features Implemented

#### Real-time Chat
- ✅ Live API calls to `/api/chat` endpoint
- ✅ Proper error handling with user-friendly messages
- ✅ Loading states with typing indicators
- ✅ Connection status in header

#### Response Display
- ✅ Formatted AI responses with proper line breaks
- ✅ Source citations with document links
- ✅ Similarity scores as percentage matches
- ✅ Processing time display
- ✅ Page and chunk references

#### Error States
- ✅ Error banner when backend is unavailable
- ✅ Visual connection status indicator
- ✅ Fallback error messages in chat

## Key Components

### App.jsx
The main chat interface component featuring:
- **API Integration**: Real chat with FastAPI backend
- **Message Management**: Dynamic message state with real responses
- **Error Handling**: Comprehensive error states and user feedback
- **Source Display**: Rich source citations with metadata
- **Performance Metrics**: Shows processing time for transparency

### API Client
```javascript
const chatWithAPI = async (message) => {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  return await response.json()
}
```

## Getting Started

### Prerequisites
- Node.js 18+ installed
- **FastAPI Backend Running** on `http://localhost:8000`
- Package manager (npm, yarn, or pnpm)

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. **IMPORTANT**: Start your FastAPI backend first:
```bash
# In the main wingman2 directory
python main.py
```

4. Start the frontend development server:
```bash
npm run dev
```

5. Open your browser and visit `http://localhost:5173`

## Usage

1. **Ask Questions**: Type Air Force-related questions about roles and responsibilities
2. **View Sources**: See which documents the AI used to generate responses
3. **Check Performance**: Monitor processing times for each query
4. **Error Recovery**: If backend goes down, you'll see clear error messages

### Example Queries
- "What does the AFOSI commander do?"
- "What are the responsibilities of a Personnel Employment specialist?"
- "Explain the roles in Air Force special investigations"

## Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint to check code quality

## Troubleshooting

### "Connection Error" in UI
- ✅ Ensure FastAPI backend is running on port 8000
- ✅ Check that `http://localhost:8000/docs` is accessible
- ✅ Verify your backend has processed some PDFs (use `/api/batch-process`)

### No AI Responses
- ✅ Check if Ollama is running (`ollama list`)
- ✅ Ensure Llama model is downloaded (`ollama pull llama3.2:3b`)
- ✅ Verify Pinecone API key is set in backend `.env`

## Architecture

```
User Input → React Frontend → FastAPI Backend → LLM (Ollama) → Vector DB (Pinecone) → Formatted Response
```

The frontend handles:
- User interface and interaction
- API communication
- Response formatting and display
- Error states and recovery

## Development Notes

- Backend must be running for full functionality
- Error states gracefully handle backend unavailability
- Source citations provide transparency into AI reasoning
- Responsive design ensures usability across all device sizes
- Performance metrics help users understand processing complexity