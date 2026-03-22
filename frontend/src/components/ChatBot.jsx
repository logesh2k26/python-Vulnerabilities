import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatBot.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
const API_KEY = import.meta.env.VITE_API_KEY || '';

function ChatBot({ vulnerability, onClose }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    // Build vulnerability context string for the backend
    const vulnContext = vulnerability
        ? `Type: ${vulnerability.type}\nDescription: ${vulnerability.description}\nSeverity: ${vulnerability.severity || 'unknown'}\nAffected Lines: ${vulnerability.affected_lines?.join(', ') || 'N/A'}\nCode Snippet:\n${vulnerability.metadata?.snippet || 'Not available'}`
        : '';

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isTyping]);

    // Fetch initial explanation on mount
    useEffect(() => {
        if (!vulnerability) return;
        fetchInitialExplanation();
    }, [vulnerability]);

    // Focus input on mount
    useEffect(() => {
        setTimeout(() => inputRef.current?.focus(), 300);
    }, []);

    const fetchInitialExplanation = async () => {
        setIsTyping(true);

        try {
            let codeSnippet = vulnerability.metadata?.snippet || 'Code context not available.';

            const response = await fetch(`${API_BASE_URL}/explain`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY,
                },
                body: JSON.stringify({
                    vulnerability_type: vulnerability.type,
                    description: vulnerability.description,
                    code_snippet: codeSnippet,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch explanation.');
            }

            const data = await response.json();
            setMessages([{ role: 'assistant', content: data.explanation }]);
        } catch (err) {
            setMessages([{
                role: 'assistant',
                content: `⚠️ **Error:** ${err.message}\n\nI couldn't analyze this vulnerability right now. You can still ask me questions about it!`
            }]);
        } finally {
            setIsTyping(false);
        }
    };

    const sendMessage = async () => {
        const text = input.trim();
        if (!text || isTyping) return;

        const userMessage = { role: 'user', content: text };
        const updatedMessages = [...messages, userMessage];
        setMessages(updatedMessages);
        setInput('');
        setIsTyping(true);

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': API_KEY,
                },
                body: JSON.stringify({
                    messages: updatedMessages.map(m => ({ role: m.role, content: m.content })),
                    vulnerability_context: vulnContext,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get response.');
            }

            const data = await response.json();
            setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `⚠️ **Error:** ${err.message}`
            }]);
        } finally {
            setIsTyping(false);
            inputRef.current?.focus();
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const suggestedQuestions = [
        "How would an attacker exploit this?",
        "Show me the fixed code",
        "What's the severity impact?",
    ];

    return (
        <div className="chatbot-overlay" onClick={onClose}>
            <div className="chatbot-panel glass-card" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="chat-header">
                    <div className="chat-header-info">
                        <div className="chat-avatar">🤖</div>
                        <div>
                            <h3 className="chat-title">Security Assistant</h3>
                            <span className="chat-subtitle">
                                Analyzing: {vulnerability?.type?.replace(/_/g, ' ') || 'Vulnerability'}
                            </span>
                        </div>
                    </div>
                    <button className="chat-close-btn" onClick={onClose} aria-label="Close">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                            <path d="M1 1L13 13M1 13L13 1" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                    </button>
                </div>

                {/* Messages area */}
                <div className="chat-messages">
                    {/* Welcome message if no messages yet and not typing */}
                    {messages.length === 0 && !isTyping && (
                        <div className="chat-welcome">
                            <div className="chat-welcome-icon">🔍</div>
                            <p>Analyzing vulnerability...</p>
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <div key={idx} className={`chat-bubble ${msg.role}`}>
                            {msg.role === 'assistant' && (
                                <div className="bubble-avatar">🤖</div>
                            )}
                            <div className="bubble-content">
                                {msg.role === 'assistant' ? (
                                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                                ) : (
                                    <p>{msg.content}</p>
                                )}
                            </div>
                        </div>
                    ))}

                    {/* Typing indicator */}
                    {isTyping && (
                        <div className="chat-bubble assistant">
                            <div className="bubble-avatar">🤖</div>
                            <div className="bubble-content">
                                <div className="typing-indicator">
                                    <span></span>
                                    <span></span>
                                    <span></span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Suggested questions — shown after first assistant message */}
                    {messages.length === 1 && messages[0].role === 'assistant' && !isTyping && (
                        <div className="chat-suggestions">
                            {suggestedQuestions.map((q, idx) => (
                                <button
                                    key={idx}
                                    className="suggestion-chip"
                                    onClick={() => {
                                        setInput(q);
                                        setTimeout(() => {
                                            setInput('');
                                            const userMsg = { role: 'user', content: q };
                                            const updated = [...messages, userMsg];
                                            setMessages(updated);
                                            setIsTyping(true);

                                            fetch(`${API_BASE_URL}/chat`, {
                                                method: 'POST',
                                                headers: {
                                                    'Content-Type': 'application/json',
                                                    'X-API-Key': API_KEY,
                                                },
                                                body: JSON.stringify({
                                                    messages: updated.map(m => ({ role: m.role, content: m.content })),
                                                    vulnerability_context: vulnContext,
                                                }),
                                            })
                                            .then(r => r.json())
                                            .then(data => {
                                                setMessages(prev => [...prev, { role: 'assistant', content: data.reply || data.detail }]);
                                            })
                                            .catch(err => {
                                                setMessages(prev => [...prev, { role: 'assistant', content: `⚠️ ${err.message}` }]);
                                            })
                                            .finally(() => setIsTyping(false));
                                        }, 0);
                                    }}
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input bar */}
                <div className="chat-input-bar">
                    <textarea
                        ref={inputRef}
                        className="chat-input"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask a follow-up question..."
                        rows={1}
                        disabled={isTyping}
                    />
                    <button
                        className={`chat-send-btn ${input.trim() && !isTyping ? 'active' : ''}`}
                        onClick={sendMessage}
                        disabled={!input.trim() || isTyping}
                        aria-label="Send message"
                    >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                            <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                            <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ChatBot;
