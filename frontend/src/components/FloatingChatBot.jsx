import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './FloatingChatBot.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
const API_KEY = import.meta.env.VITE_API_KEY || 'test-secret-key-12345';

function FloatingChatBot() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isTyping]);

    // Focus input when chat opens
    useEffect(() => {
        if (isOpen) {
            setTimeout(() => inputRef.current?.focus(), 300);
        }
    }, [isOpen]);

    const toggleChat = () => {
        setIsOpen(prev => !prev);
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
                    vulnerability_context: '',
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
        "How to prevent SQL injection?",
        "What is path traversal?",
        "Explain command injection",
    ];

    return (
        <>
            {/* Floating Chat Panel */}
            <div className={`floating-chat-panel glass-card ${isOpen ? 'open' : ''}`}>
                {/* Header */}
                <div className="fc-header">
                    <div className="fc-header-info">
                        <div className="fc-avatar">🤖</div>
                        <div>
                            <h3 className="fc-title">Security Assistant</h3>
                            <span className="fc-subtitle">Ask anything about security</span>
                        </div>
                    </div>
                    <button className="fc-close-btn" onClick={toggleChat} aria-label="Close chat">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                            <path d="M1 1L13 13M1 13L13 1" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                    </button>
                </div>

                {/* Messages area */}
                <div className="fc-messages">
                    {/* Welcome message if no messages */}
                    {messages.length === 0 && !isTyping && (
                        <div className="fc-welcome">
                            <div className="fc-welcome-icon">🛡️</div>
                            <h4 className="fc-welcome-title">Security Assistant</h4>
                            <p className="fc-welcome-text">
                                Ask me anything about Python security, vulnerabilities, or best practices.
                            </p>
                            <div className="fc-quick-actions">
                                {suggestedQuestions.map((q, idx) => (
                                    <button
                                        key={idx}
                                        className="fc-suggestion-chip"
                                        onClick={() => {
                                            setInput(q);
                                            setTimeout(() => {
                                                const userMsg = { role: 'user', content: q };
                                                const updated = [userMsg];
                                                setMessages(updated);
                                                setInput('');
                                                setIsTyping(true);

                                                fetch(`${API_BASE_URL}/chat`, {
                                                    method: 'POST',
                                                    headers: {
                                                        'Content-Type': 'application/json',
                                                        'X-API-Key': API_KEY,
                                                    },
                                                    body: JSON.stringify({
                                                        messages: updated.map(m => ({ role: m.role, content: m.content })),
                                                        vulnerability_context: '',
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
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <div key={idx} className={`fc-bubble ${msg.role}`}>
                            {msg.role === 'assistant' && (
                                <div className="fc-bubble-avatar">🤖</div>
                            )}
                            <div className="fc-bubble-content">
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
                        <div className="fc-bubble assistant">
                            <div className="fc-bubble-avatar">🤖</div>
                            <div className="fc-bubble-content">
                                <div className="fc-typing-indicator">
                                    <span></span>
                                    <span></span>
                                    <span></span>
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input bar */}
                <div className="fc-input-bar">
                    <textarea
                        ref={inputRef}
                        className="fc-input"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask about security..."
                        rows={1}
                        disabled={isTyping}
                    />
                    <button
                        className={`fc-send-btn ${input.trim() && !isTyping ? 'active' : ''}`}
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

            {/* Floating Action Button */}
            <button
                className={`floating-chat-fab ${isOpen ? 'active' : ''}`}
                onClick={toggleChat}
                aria-label="Open chatbot"
                id="floating-chat-toggle"
            >
                <span className="fab-icon-chat">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
                        <path d="M21 11.5C21 16.75 16.75 21 11.5 21C9.8 21 8.2 20.55 6.8 19.75L2 21L3.25 16.2C2.45 14.8 2 13.2 2 11.5C2 6.25 6.25 2 11.5 2C16.75 2 21 6.25 21 11.5Z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                        <path d="M8 10H8.01" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                        <path d="M12 10H12.01" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                        <path d="M16 10H16.01" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                    </svg>
                </span>
                <span className="fab-icon-close">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                        <path d="M18 6L6 18" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
                        <path d="M6 6L18 18" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"/>
                    </svg>
                </span>
            </button>

            {/* Notification badge pulse ring */}
            {!isOpen && messages.length === 0 && (
                <div className="fab-pulse-ring"></div>
            )}
        </>
    );
}

export default FloatingChatBot;
