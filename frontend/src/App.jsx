import React, { useState, useRef, useEffect } from 'react'

// API Configuration
const API_BASE_URL = 'http://localhost:8000'

// API Client Function
const chatWithAPI = async (message) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('API Error:', error)
    throw error
  }
}

// Initial welcome message
const initialMessages = [
  {
    id: 1,
    type: 'agent',
    content: 'Hello! I\'m your Air Force RAG Assistant. I can help you find information about roles, responsibilities, and compliance from Air Force manuals.\n\n**What I Can Help You With:**\n\n1. Air Force roles and responsibilities\n2. Command structure and authority\n3. Personnel duties and procedures\n4. Compliance requirements\n5. Organizational responsibilities\n\n**Sample Questions:**\n• What does the AFOSI commander do?\n• Personnel Employment specialist responsibilities\n• Air Force investigation procedures\n\nWhat would you like to know?',
    timestamp: new Date(),
    sources: []
  }
]

function App() {
  const [messages, setMessages] = useState(initialMessages)
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!inputMessage.trim() || isLoading) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    const currentMessage = inputMessage.trim()
    setInputMessage('')
    setIsLoading(true)
    setError(null)

    try {
      // Call the real API
      const apiResponse = await chatWithAPI(currentMessage)
      
      // Format sources for the UI
      const formattedSources = apiResponse.sources?.map(source => ({
        title: source.doc_type && source.document ? 
          `${source.doc_type} - ${source.document.split('/').pop().replace('.pdf', '')}` : 
          source.document || 'Unknown Document',
        url: source.document || '#',
        page: source.page || 'Unknown',
        score: source.similarity_score || 0,
        chunk_id: source.chunk_id
      })) || []

      const agentMessage = {
        id: Date.now() + 1,
        type: 'agent',
        content: apiResponse.response,
        timestamp: new Date(),
        sources: formattedSources,
        processingTime: apiResponse.processing_time
      }

      setMessages(prev => [...prev, agentMessage])
      
    } catch (error) {
      console.error('Failed to get response:', error)
      setError('Failed to connect to the AI assistant. Please check if the backend server is running.')
      
      // Show error message to user
      const errorMessage = {
        id: Date.now() + 1,
        type: 'agent',
        content: '⚠️ I\'m having trouble connecting to my knowledge base right now. Please make sure the backend server is running on port 8000 and try again.',
        timestamp: new Date(),
        sources: [],
        isError: true
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit'
    })
  }

  // Function to format and beautify LLM responses
  const formatLLMResponse = (content) => {
    if (!content) return null

    const lines = content.split('\n')
    const formattedContent = []
    let currentSection = null
    let listItems = []

    lines.forEach((line, index) => {
      const trimmedLine = line.trim()
      
      if (!trimmedLine) {
        // Empty line - add spacing
        if (listItems.length > 0) {
          formattedContent.push(
            <div key={`list-${index}`} className="mb-4">
              <ul className="space-y-2">
                {listItems.map((item, itemIndex) => (
                  <li key={itemIndex} className="flex items-start gap-3">
                    <span className="text-green-600 text-lg mt-0.5 flex-shrink-0">✓</span>
                    <span className="flex-1">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )
          listItems = []
        }
        if (currentSection) {
          formattedContent.push(<div key={`section-${index}`} className="mb-4">{currentSection}</div>)
          currentSection = null
        }
        formattedContent.push(<div key={`space-${index}`} className="h-2"></div>)
        return
      }

      // Check if line is a header/title (starts with ** or contains keywords like "Role:", "Responsibilities:", etc.)
      if (trimmedLine.startsWith('**') && trimmedLine.endsWith('**')) {
        // Bold header
        const headerText = trimmedLine.slice(2, -2)
        if (listItems.length > 0) {
          formattedContent.push(
            <div key={`list-before-header-${index}`} className="mb-4">
              <ul className="space-y-2">
                {listItems.map((item, itemIndex) => (
                  <li key={itemIndex} className="flex items-start gap-3">
                    <span className="text-green-600 text-lg mt-0.5 flex-shrink-0">✓</span>
                    <span className="flex-1">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )
          listItems = []
        }
        if (currentSection) {
          formattedContent.push(<div key={`section-before-header-${index}`} className="mb-4">{currentSection}</div>)
        }
        
        formattedContent.push(
          <div key={`header-${index}`} className="mb-3">
            <h3 className="text-lg font-bold text-blue-700 border-l-4 border-blue-500 pl-3 py-1 bg-blue-50/50 rounded-r">
              {headerText}
            </h3>
          </div>
        )
        currentSection = null
        return
      }

      // Check for numbered lists (1., 2., etc.) or bullet points
      const numberedMatch = trimmedLine.match(/^(\d+)\.\s*(.+)/)
      const bulletMatch = trimmedLine.match(/^[-•*]\s*(.+)/)
      
      if (numberedMatch || bulletMatch) {
        const itemText = numberedMatch ? numberedMatch[2] : bulletMatch[1]
        listItems.push(itemText)
        return
      }

      // Check for role/responsibility keywords
      const roleKeywords = ['Role:', 'Responsibility:', 'Duties:', 'Authority:', 'Command:', 'Position:']
      const isRoleHeader = roleKeywords.some(keyword => trimmedLine.toLowerCase().includes(keyword.toLowerCase()))
      
      if (isRoleHeader) {
        if (listItems.length > 0) {
          formattedContent.push(
            <div key={`list-before-role-${index}`} className="mb-4">
              <ul className="space-y-2">
                {listItems.map((item, itemIndex) => (
                  <li key={itemIndex} className="flex items-start gap-3">
                    <span className="text-green-600 text-lg mt-0.5 flex-shrink-0">✓</span>
                    <span className="flex-1">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )
          listItems = []
        }
        if (currentSection) {
          formattedContent.push(<div key={`section-before-role-${index}`} className="mb-3">{currentSection}</div>)
        }
        
        formattedContent.push(
          <div key={`role-${index}`} className="mb-3">
            <div className="flex items-center gap-2 font-semibold text-blue-800 bg-blue-50 px-3 py-2 rounded-lg border-l-4 border-blue-400">
              <span className="text-blue-600">🎖️</span>
              <span>{trimmedLine}</span>
            </div>
          </div>
        )
        currentSection = null
        return
      }

      // Regular paragraph text
      if (currentSection) {
        currentSection = <>{currentSection} {trimmedLine}</>
      } else {
        currentSection = <span>{trimmedLine}</span>
      }
    })

    // Handle remaining content
    if (listItems.length > 0) {
      formattedContent.push(
        <div key="final-list" className="mb-4">
          <ul className="space-y-2">
            {listItems.map((item, itemIndex) => (
              <li key={itemIndex} className="flex items-start gap-3">
                <span className="text-green-600 text-lg mt-0.5 flex-shrink-0">✓</span>
                <span className="flex-1">{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )
    }
    
    if (currentSection) {
      formattedContent.push(<div key="final-section" className="mb-2">{currentSection}</div>)
    }

    return formattedContent.length > 0 ? formattedContent : <span>{content}</span>
  }

  const suggestionQuestions = [
    'What does the AFOSI commander do?',
    'What are the responsibilities of a Personnel Employment specialist?',
    'Explain the roles in Air Force special investigations',
    'Show me the duties of a squadron commander'
  ]

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 font-sans">
      {/* Header */}
      <header className="glass border-b border-white/20 p-6 shadow-2xl">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white text-xl shadow-lg">
              🇺🇸
            </div>
      <div>
              <h1 className="text-2xl font-bold text-gray-800 text-shadow">
                Air Force RAG Assistant
              </h1>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <div className={`w-2 h-2 rounded-full animate-pulse ${
                  error ? 'bg-red-500' : 'bg-green-500'
                }`}></div>
                <span>
                  {error ? 'Connection Error • Check backend server' : 'Online • Ready to help with Air Force documentation'}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex gap-8">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {messages.filter(m => m.type === 'agent').length}
              </div>
              <div className="text-xs text-gray-500 uppercase tracking-wide">
                Responses
              </div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">10+</div>
              <div className="text-xs text-gray-500 uppercase tracking-wide">
                Manuals
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 mx-6 mt-4 rounded-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <span className="text-red-500 text-xl">⚠️</span>
      </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">
                <strong>Connection Error:</strong> {error}
              </p>
              <p className="text-xs text-red-600 mt-1">
                Make sure your backend server is running on <code className="bg-red-100 px-1 rounded">http://localhost:8000</code>
        </p>
      </div>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <main className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto scrollbar-custom p-6 max-w-4xl mx-auto w-full">
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`flex gap-3 mb-6 animate-slideIn ${
                message.type === 'user' ? 'flex-row-reverse' : ''
              }`}
            >
              {/* Avatar */}
              <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg flex-shrink-0 mt-1 shadow-lg ${
                message.type === 'agent' 
                  ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white' 
                  : 'bg-gradient-to-br from-green-500 to-green-600 text-white'
              }`}>
                {message.type === 'agent' ? '🤖' : '👤'}
              </div>

              {/* Message Content */}
              <div className={`flex-1 max-w-[70%] ${
                message.type === 'user' ? 'flex flex-col items-end' : ''
              }`}>
                <div className={`rounded-2xl p-4 shadow-xl ${
                  message.type === 'agent'
                    ? message.isError 
                      ? 'bg-red-50 border border-red-200' 
                      : 'glass border border-white/20'
                    : 'bg-gradient-to-br from-blue-500 to-blue-600 text-white'
                }`}>
                  <div className={`leading-relaxed ${
                    message.type === 'agent' 
                      ? message.isError 
                        ? 'text-red-800' 
                        : 'text-gray-800' 
                      : 'text-white'
                  }`}>
                    {message.type === 'agent' && !message.isError ? (
                      // Use beautified formatting for agent responses
                      <div className="space-y-2">
                        {formatLLMResponse(message.content)}
                      </div>
                    ) : (
                      // Use simple formatting for user messages and errors
                      message.content.split('\n').map((line, index) => (
                        <React.Fragment key={index}>
                          {line}
                          {index < message.content.split('\n').length - 1 && <br />}
                        </React.Fragment>
                      ))
                    )}
                  </div>
                  
                  {/* Processing Time */}
                  {message.processingTime && (
                    <div className="mt-2 text-xs text-gray-500">
                      ⚡ Processed in {message.processingTime}s
                    </div>
                  )}
                  
                  {/* Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-6 pt-4 border-t border-gray-200/30">
                      <div className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-3">
                        <span className="text-blue-600">📚</span>
                        <span>Reference Sources ({message.sources.length})</span>
                      </div>
                      <div className="space-y-3">
                        {message.sources.map((source, index) => (
                          <div key={index} className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-3 hover:shadow-md transition-all duration-200">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <a 
                                  href={source.url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-blue-700 hover:text-blue-800 hover:underline transition-colors font-semibold text-sm flex items-center gap-2"
                                >
                                  <span className="text-blue-600">📄</span>
                                  {source.title}
                                </a>
                                <div className="text-xs text-gray-600 mt-1 flex items-center gap-3">
                                  <span className="flex items-center gap-1">
                                    <span className="w-1 h-1 bg-gray-400 rounded-full"></span>
                                    Page: {source.page}
                                  </span>
                                  {source.chunk_id !== undefined && (
                                    <span className="flex items-center gap-1">
                                      <span className="w-1 h-1 bg-gray-400 rounded-full"></span>
                                      Section: {source.chunk_id}
                                    </span>
                                  )}
                                </div>
                              </div>
                              {source.score && (
                                <div className="ml-3 flex flex-col items-end">
                                  <div className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">
                                    {Math.round(source.score * 100)}% match
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Timestamp */}
                <div className={`text-lg text-black mt-2 ${
                  message.type === 'user' ? 'text-right' : 'text-left'
                }`}>
                  {formatTimestamp(message.timestamp)}
                </div>
              </div>
            </div>
          ))}
          
          {/* Loading indicator */}
          {isLoading && (
            <div className="flex gap-3 mb-6 animate-slideIn">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white text-lg shadow-lg">
                🤖
              </div>
              <div className="glass rounded-2xl p-4 border border-white/20 shadow-xl">
                <div className="flex gap-1 items-center">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Form */}
      <footer className="glass border-t border-white/20 p-6 shadow-2xl">
        <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto">
          <div className="flex gap-3 mb-4">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Ask about Air Force roles, responsibilities, or compliance..."
              className="flex-1 px-5 py-3 border-2 border-gray-200/20 rounded-full outline-none text-base glass backdrop-blur-sm transition-all duration-200 focus:border-blue-500 focus:shadow-lg focus:bg-white/95 disabled:opacity-60 disabled:cursor-not-allowed"
              disabled={isLoading}
            />
            <button 
              type="submit" 
              className="w-12 h-12 rounded-full border-none bg-gradient-to-br from-blue-500 to-blue-600 text-white text-lg cursor-pointer transition-all duration-200 flex items-center justify-center shadow-lg hover:transform hover:-translate-y-0.5 hover:shadow-xl disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
              disabled={!inputMessage.trim() || isLoading}
            >
              {isLoading ? '⏳' : '📤'}
            </button>
          </div>
          
          {/* Suggestion Chips */}
          <div className="flex gap-3 flex-wrap justify-center">
            {suggestionQuestions.map((question, index) => (
              <button 
                key={index}
                type="button" 
                className="px-4 py-2 border border-gray-200/30 rounded-full glass backdrop-blur-sm text-gray-700 text-sm cursor-pointer transition-all duration-200 hover:bg-blue-100/20 hover:text-blue-700 hover:transform hover:-translate-y-0.5"
                onClick={() => setInputMessage(question)}
              >
                {question.length > 30 ? question.substring(0, 30) + '...' : question}
              </button>
            ))}
          </div>
        </form>
      </footer>
    </div>
  )
}

export default App