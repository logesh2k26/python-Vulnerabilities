import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
const API_KEY = import.meta.env.VITE_API_KEY || '';

function AIAssistantPage() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isTyping]);

    const sendMessage = async (textOverride = null) => {
        const text = textOverride || input.trim();
        if (!text || isTyping) return;

        const userMessage = { role: 'user', content: text, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) };
        const updatedMessages = [...messages, userMessage];
        setMessages(updatedMessages);
        if (!textOverride) setInput('');
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
            setMessages(prev => [...prev, { role: 'assistant', content: data.reply, timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }]);
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `⚠️ **Error:** ${err.message}`,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
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
        { text: "Explain the vulnerability", icon: "info" },
        { text: "Generate remediation code", icon: "build" },
        { text: "View similar incidents", icon: "history" }
    ];

    return (
        <div className="ap-container">
            {/* Hero Header Section */}
            <section className="ap-hero">
                <div className="ap-hero-grid">
                    <div className="ap-hero-content">
                        <span className="ap-badge">Security Intelligence</span>
                        <h1 className="ap-title">
                            Architectural <br/>
                            <span className="ap-title-italic">Remediation.</span>
                        </h1>
                        <p className="ap-subtitle">
                            Navigate the complexities of cybersecurity with an assistant that doesn't just find vulnerabilities—it explains their DNA and guides your recovery.
                        </p>
                    </div>

                </div>
            </section>

            {/* AI Assistant Interface */}
            <section className="ap-interface-grid">
                {/* Chat Window */}
                <div className="ap-chat-column">
                    <div className="ap-chat-window">
                        <div className="ap-messages-area">
                            {/* Initial Message */}
                            <div className="ap-message ai">
                                <div className="ap-avatar ai">
                                    <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
                                </div>
                                <div className="ap-message-content">
                                    <span className="ap-message-meta">SkyDrop AI • Just now</span>
                                    <div className="ap-message-bubble">
                                        Hello! I'm here to help you secure your Python applications. You can ask me about vulnerability remediation, secure coding patterns, or details from your latest scan.
                                    </div>
                                </div>
                            </div>

                            {/* Chat History */}
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`ap-message ${msg.role}`}>
                                    <div className={`ap-avatar ${msg.role}`}>
                                        <span className="material-symbols-outlined" style={msg.role === 'assistant' ? { fontVariationSettings: "'FILL' 1" } : {}}>
                                            {msg.role === 'assistant' ? 'auto_awesome' : 'person'}
                                        </span>
                                    </div>
                                    <div className="ap-message-content">
                                        <span className="ap-message-meta">{msg.role === 'assistant' ? 'SkyDrop AI' : 'You'} • {msg.timestamp}</span>
                                        <div className="ap-message-bubble">
                                            {msg.role === 'assistant' ? (
                                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                                            ) : (
                                                msg.content
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}

                            {isTyping && (
                                <div className="ap-message ai">
                                    <div className="ap-avatar ai">
                                        <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>auto_awesome</span>
                                    </div>
                                    <div className="ap-typing">
                                        <span className="ap-dot"></span>
                                        <span className="ap-dot"></span>
                                        <span className="ap-dot"></span>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Suggestions Area */}
                        {!messages.length && (
                            <div className="ap-suggestions">
                                {suggestedQuestions.map((q, idx) => (
                                    <button 
                                        key={idx} 
                                        className="ap-suggestion-btn"
                                        onClick={() => sendMessage(q.text)}
                                    >
                                        <span className="material-symbols-outlined">{q.icon}</span>
                                        {q.text}
                                    </button>
                                ))}
                            </div>
                        )}

                        {/* Input Area */}
                        <div className="ap-input-container">
                            <div className="ap-input-wrapper">
                                <span className="material-symbols-outlined ap-input-icon">terminal</span>
                                <input 
                                    ref={inputRef}
                                    type="text" 
                                    className="ap-input"
                                    placeholder="Ask about remediation, patches, or security docs..."
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    disabled={isTyping}
                                />
                                <button 
                                    className="ap-send-btn"
                                    onClick={() => sendMessage()}
                                    disabled={!input.trim() || isTyping}
                                >
                                    <span className="material-symbols-outlined">send</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Bento Column */}
                <div className="ap-bento-column">
                    <div className="ap-card capabilities">
                        <h3 className="ap-card-title">Capabilities</h3>
                        <div className="ap-feature-list">
                            <div className="ap-feature-item">
                                <div className="ap-feature-header">
                                    <span className="material-symbols-outlined">verified_user</span>
                                    <strong>Remediation Advice</strong>
                                </div>
                                <p>Step-by-step guidance to patch critical vulnerabilities in production.</p>
                            </div>
                            <div className="ap-feature-item">
                                <div className="ap-feature-header">
                                    <span className="material-symbols-outlined">menu_book</span>
                                    <strong>Code Context</strong>
                                </div>
                                <p>AI understands your specific library versions and architecture patterns.</p>
                            </div>
                            <div className="ap-feature-item">
                                <div className="ap-feature-header">
                                    <span className="material-symbols-outlined">security</span>
                                    <strong>Zero-Day Intel</strong>
                                </div>
                                <p>Real-time alerts for newly discovered exploits in your stack.</p>
                            </div>
                        </div>
                    </div>

                    <div className="ap-card visualization">
                        <div className="ap-visual-content">
                            <h3 className="ap-visual-title">Network Visualization</h3>
                            <p className="ap-visual-desc">Visualize attack vectors in 3D using AI projection.</p>
                        </div>
                        <img 
                            src="https://lh3.googleusercontent.com/aida-public/AB6AXuC9w7Ju14K7I87WCteVgkHGicZQ9jRXRgvZw7lQ_nFXaadOixTrhguW9RjPBtoJN_ET4U5_0fs4kviPU-C4AAkVmph3yZWTbjDGh0iTDxm7vPEml9qr-K2NFWS4lcXGFtHjgMuhACQ25CouZBHvU6nJa5oQ9DYO-j3pykmJhOv-v4H1v4g22TUSwWCr3VVQ1OITcrDW2epNcYhE8wGblW0Oo0n6ykdT2CNa2omUb21E6YdOD6UufR1gwm-dR1k_Vw-ww1Tzr9QeTwHs" 
                            alt="Data visualization" 
                            className="ap-visual-img"
                        />
                    </div>
                </div>
            </section>

            {/* Bottom Status Bar */}
            <footer className="ap-footer">
                <div className="ap-footer-left">
                    <div className="ap-status-item">
                        <label>Encryption</label>
                        <span>AES-256 E2E</span>
                    </div>
                    <div className="ap-status-item">
                        <label>Model</label>
                        <span>SkyDrop-L4 Secure</span>
                    </div>
                </div>
                <div className="ap-system-status">
                    <span className="ap-status-dot pulse"></span>
                    <span>System Status: Optimal</span>
                </div>
            </footer>

            <style>{`
                .ap-container {
                    padding: 60px 80px;
                    max-width: 1300px;
                    margin: 0 auto;
                }

                /* Hero Header */
                .ap-hero { margin-bottom: 60px; }
                .ap-hero-grid {
                    display: grid;
                    grid-template-columns: 1fr auto;
                    align-items: flex-end;
                    gap: 40px;
                }
                .ap-badge {
                    display: inline-block;
                    padding: 5px 14px;
                    background: #e0e0ff;
                    color: #4c56af;
                    font-size: 10px;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.12em;
                    border-radius: 999px;
                    margin-bottom: 24px;
                }
                .ap-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 56px;
                    font-weight: 800;
                    line-height: 1.1;
                    letter-spacing: -0.02em;
                    color: #29343a;
                    margin-bottom: 24px;
                }
                .ap-title-italic {
                    color: #4c56af;
                    font-style: italic;
                }
                .ap-subtitle {
                    font-size: 18px;
                    color: #415660;
                    max-width: 580px;
                    line-height: 1.6;
                }


                /* Main Grid */
                .ap-interface-grid {
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 40px;
                }

                /* Chat Window */
                .ap-chat-window {
                    background: #ffffff;
                    border-radius: 24px;
                    box-shadow: 0 16px 48px -12px rgba(41, 52, 58, 0.08);
                    display: flex;
                    flex-direction: column;
                    min-height: 600px;
                    overflow: hidden;
                    border: 1px solid rgba(168, 179, 187, 0.1);
                }
                .ap-messages-area {
                    flex: 1;
                    padding: 32px;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 32px;
                }
                .ap-message {
                    display: flex;
                    gap: 20px;
                }
                .ap-avatar {
                    width: 44px;
                    height: 44px;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-shrink: 0;
                }
                .ap-avatar.ai {
                    background: #e0e0ff;
                }
                .ap-avatar.ai .material-symbols-outlined {
                    color: #4c56af;
                    font-variation-settings: 'FILL' 1;
                }
                .ap-avatar.user {
                    background: #f0f4f8;
                }
                .ap-avatar.user .material-symbols-outlined {
                    color: #717c84;
                }
                .ap-message-content {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                .ap-message-meta {
                    font-size: 10px;
                    font-weight: 700;
                    color: #717c84;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                }
                .ap-message-bubble {
                    font-size: 16px;
                    line-height: 1.5;
                    color: #29343a;
                    max-width: 90%;
                }
                .ap-message.user .ap-avatar { order: 2; }
                .ap-message.user .ap-message-content { order: 1; align-items: flex-end; text-align: right; margin-left: auto; }
                .ap-message.user .ap-message-bubble { background: #f7f9fc; padding: 12px 20px; border-radius: 16px 4px 16px 16px; }

                /* Typing Indicator */
                .ap-typing {
                    display: flex;
                    gap: 4px;
                    padding: 12px 16px;
                    background: #f7f9fc;
                    border-radius: 12px;
                    width: fit-content;
                }
                .ap-dot {
                    width: 6px;
                    height: 6px;
                    background: #4c56af;
                    border-radius: 50%;
                    animation: ap-blink 1.4s infinite both;
                    opacity: 0.4;
                }
                .ap-dot:nth-child(2) { animation-delay: 0.2s; }
                .ap-dot:nth-child(3) { animation-delay: 0.4s; }
                @keyframes ap-blink {
                    0%, 80%, 100% { opacity: 0.4; transform: scale(0.9); }
                    40% { opacity: 1; transform: scale(1.1); }
                }

                /* Suggestions */
                .ap-suggestions {
                    padding: 0 32px 24px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 12px;
                }
                .ap-suggestion-btn {
                    padding: 10px 18px;
                    background: white;
                    border: 1px solid rgba(168, 179, 187, 0.2);
                    border-radius: 12px;
                    font-size: 13px;
                    font-weight: 600;
                    color: #415660;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .ap-suggestion-btn:hover {
                    background: #f7f9fc;
                    border-color: #4c56af;
                    color: #4c56af;
                }
                .ap-suggestion-btn .material-symbols-outlined {
                    font-size: 18px;
                    color: #4c56af;
                }

                /* Input Area */
                .ap-input-container {
                    padding: 24px 32px 32px;
                    border-top: 1px solid rgba(168, 179, 187, 0.08);
                }
                .ap-input-wrapper {
                    position: relative;
                    display: flex;
                    align-items: center;
                }
                .ap-input-icon {
                    position: absolute;
                    left: 18px;
                    color: #717c84;
                    font-size: 20px;
                }
                .ap-input {
                    width: 100%;
                    background: #f0f4f8;
                    border: none;
                    border-radius: 16px;
                    padding: 18px 60px 18px 52px;
                    font-size: 15px;
                    color: #29343a;
                    outline: none;
                    font-weight: 500;
                    transition: all 0.2s;
                }
                .ap-input:focus {
                    background: #e1e9f0;
                    box-shadow: 0 0 0 3px rgba(76, 86, 175, 0.1);
                }
                .ap-send-btn {
                    position: absolute;
                    right: 18px;
                    background: #4c56af;
                    color: white;
                    border: none;
                    width: 36px;
                    height: 36px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .ap-send-btn:hover { background: #4049a2; }
                .ap-send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

                /* Bento Column */
                .ap-bento-column {
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                }
                .ap-card {
                    background: #f0f4f8;
                    border-radius: 24px;
                    padding: 28px;
                    position: relative;
                    overflow: hidden;
                }
                .ap-card-title {
                    font-size: 11px;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: #717c84;
                    margin-bottom: 24px;
                }
                .ap-feature-list {
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                }
                .ap-feature-item {
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                }
                .ap-feature-header {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                .ap-feature-header .material-symbols-outlined {
                    color: #4c56af;
                    font-size: 20px;
                }
                .ap-feature-header strong {
                    font-size: 14px;
                    color: #29343a;
                }
                .ap-feature-item p {
                    font-size: 12px;
                    color: #415660;
                    padding-left: 32px;
                    line-height: 1.5;
                }

                .ap-card.visualization {
                    padding: 0;
                    min-height: 320px;
                    display: flex;
                    flex-direction: column;
                }
                .ap-visual-content {
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    padding: 28px;
                    background: linear-gradient(to top, rgba(41, 52, 58, 0.95), transparent);
                    z-index: 2;
                    color: white;
                }
                .ap-visual-title { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
                .ap-visual-desc { font-size: 12px; opacity: 0.8; }
                .ap-visual-img {
                    width: 100%;
                    height: 100%;
                    object-cover: cover;
                    position: absolute;
                    top: 0;
                    left: 0;
                    transition: transform 0.6s ease;
                }
                .ap-card.visualization:hover .ap-visual-img { transform: scale(1.1); }

                /* Footer */
                .ap-footer {
                    margin-top: 80px;
                    padding-top: 40px;
                    border-top: 1px solid rgba(168, 179, 187, 0.1);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .ap-footer-left {
                    display: flex;
                    gap: 40px;
                }
                .ap-status-item {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }
                .ap-status-item label {
                    font-size: 10px;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.15em;
                    color: #717c84;
                }
                .ap-status-item span {
                    font-size: 13px;
                    font-weight: 600;
                    color: #415660;
                }
                .ap-system-status {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    background: #d1e4fe;
                    padding: 8px 16px;
                    border-radius: 999px;
                }
                .ap-system-status span:last-child {
                    font-size: 11px;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    color: #415368;
                }
                .ap-status-dot {
                    width: 8px;
                    height: 8px;
                    background: #4c56af;
                    border-radius: 50%;
                }
                .ap-status-dot.pulse {
                    box-shadow: 0 0 0 rgba(76, 86, 175, 0.4);
                    animation: ap-pulse 2s infinite;
                }
                @keyframes ap-pulse {
                    0% { box-shadow: 0 0 0 0 rgba(76, 86, 175, 0.7); }
                    70% { box-shadow: 0 0 0 10px rgba(76, 86, 175, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(76, 86, 175, 0); }
                }
            `}</style>
        </div>
    );
}

export default AIAssistantPage;
