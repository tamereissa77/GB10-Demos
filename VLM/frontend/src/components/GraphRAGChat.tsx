import React, { useState } from 'react';
import './GraphRAGChat.css';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    sources?: {
        vectorChunks?: Array<{ text: string; score: number }>;
        graphTriples?: Array<{ subject: string; predicate: string; object: string }>;
    };
    metadata?: {
        vectorResultCount?: number;
        graphTripleCount?: number;
        durationSeconds?: number;
    };
}

const GraphRAGChat: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [showSources, setShowSources] = useState<number | null>(null);

    const sendQuery = async () => {
        if (!input.trim() || loading) return;

        const userMsg: Message = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const resp = await fetch('/chat-api/api/graphrag-query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: input, topK: 5 }),
            });

            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            const assistantMsg: Message = {
                role: 'assistant',
                content: data.answer || 'No answer generated.',
                sources: data.sources,
                metadata: data.metadata,
            };
            setMessages(prev => [...prev, assistantMsg]);
        } catch (err: any) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${err.message}`,
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    };

    return (
        <div className="graphrag-chat">
            <div className="chat-header">
                <h3>📊 GraphRAG Q&A</h3>
                <span className="chat-subtitle">Ask questions about your documents</span>
            </div>

            <div className="chat-messages">
                {messages.length === 0 && (
                    <div className="chat-empty">
                        <p>Upload a document first, then ask questions here.</p>
                        <p className="chat-examples">Try: <em>"What are the total assets?"</em> or <em>"Summarize the key financial data"</em></p>
                    </div>
                )}
                {messages.map((msg, idx) => (
                    <div key={idx} className={`chat-message ${msg.role}`}>
                        <div className="message-content">
                            {msg.content}
                        </div>
                        {msg.role === 'assistant' && msg.metadata && (
                            <div className="message-meta">
                                <span>⏱ {msg.metadata.durationSeconds}s</span>
                                <span>📄 {msg.metadata.vectorResultCount} chunks</span>
                                <span>🔗 {msg.metadata.graphTripleCount} triples</span>
                                <button
                                    className="sources-toggle"
                                    onClick={() => setShowSources(showSources === idx ? null : idx)}
                                >
                                    {showSources === idx ? 'Hide sources' : 'Show sources'}
                                </button>
                            </div>
                        )}
                        {showSources === idx && msg.sources && (
                            <div className="message-sources">
                                {msg.sources.graphTriples && msg.sources.graphTriples.length > 0 && (
                                    <div className="source-section">
                                        <h4>🔗 Graph Triples</h4>
                                        <ul>
                                            {msg.sources.graphTriples.slice(0, 10).map((t, i) => (
                                                <li key={i}>{t.subject} → {t.predicate} → {t.object}</li>
                                            ))}
                                        </ul>
                                    </div>
                                )}
                                {msg.sources.vectorChunks && msg.sources.vectorChunks.length > 0 && (
                                    <div className="source-section">
                                        <h4>📄 Text Chunks</h4>
                                        {msg.sources.vectorChunks.map((c, i) => (
                                            <div key={i} className="text-chunk">
                                                <span className="chunk-score">Score: {c.score.toFixed(3)}</span>
                                                <p>{c.text.substring(0, 200)}...</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ))}
                {loading && (
                    <div className="chat-message assistant loading">
                        <div className="typing-indicator">
                            <span></span><span></span><span></span>
                        </div>
                    </div>
                )}
            </div>

            <div className="chat-input-area">
                <input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a question about your documents..."
                    disabled={loading}
                />
                <button onClick={sendQuery} disabled={loading || !input.trim()}>
                    {loading ? '...' : 'Ask'}
                </button>
            </div>
        </div>
    );
};

export default GraphRAGChat;
