import { useState } from 'react'
import './App.css'
import FileUpload from './components/FileUpload'
import ResultViewer from './components/ResultViewer'
import GraphRAGChat from './components/GraphRAGChat'

function App() {
  const mode = (import.meta.env.VITE_APP_MODE || 'extraction') as 'extraction' | 'chat'
  const [result, setResult] = useState<any>(null)

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>{mode === 'chat' ? 'GraphRAG Chat' : 'Arabic OCR & Layout Analysis'}</h1>
        <p>{mode === 'chat' ? 'Query your knowledge graph' : 'Powered by datalab-to/chandra'}</p>
      </header>

      <main className="app-main">
        {mode === 'chat' ? (
          <GraphRAGChat />
        ) : (
          <>
            <FileUpload onUploadSuccess={setResult} />
            <ResultViewer data={result} />
          </>
        )}
      </main>
    </div>
  )
}

export default App
