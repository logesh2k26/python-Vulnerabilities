import { useState, useCallback } from 'react'
import Dashboard from './components/Dashboard'
import './index.css'

function App() {
    const [analysisResult, setAnalysisResult] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState(null)

    const analyzeCode = useCallback(async (code, filename = 'code.py') => {
        setIsLoading(true)
        setError(null)

        try {
            const response = await fetch('/api/v1/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: code, filename })
            })

            if (!response.ok) {
                const data = await response.json()
                throw new Error(data.detail || 'Analysis failed')
            }

            const result = await response.json()
            setAnalysisResult(result)
        } catch (err) {
            setError(err.message)
            setAnalysisResult(null)
        } finally {
            setIsLoading(false)
        }
    }, [])

    const analyzeFiles = useCallback(async (files) => {
        setIsLoading(true)
        setError(null)

        const fileContents = await Promise.all(
            files.map(async (file) => ({
                filename: file.name,
                content: await file.text()
            }))
        )

        try {
            const response = await fetch('/api/v1/analyze/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ files: fileContents })
            })

            if (!response.ok) throw new Error('Batch analysis failed')

            const result = await response.json()
            if (result.results && result.results.length > 0) {
                setAnalysisResult(result.results[0])
            }
        } catch (err) {
            setError(err.message)
        } finally {
            setIsLoading(false)
        }
    }, [])

    return (
        <div className="app-container">
            <header className="header glass-card">
                <div className="logo">
                    <div className="logo-icon">🛡️</div>
                    <span className="logo-text">PyVulnDetect</span>
                </div>
                <div className="header-status">
                    <span className="status-dot"></span>
                    <span>AI Engine Ready</span>
                </div>
            </header>

            <Dashboard
                analysisResult={analysisResult}
                isLoading={isLoading}
                error={error}
                onAnalyze={analyzeCode}
                onUpload={analyzeFiles}
            />

            {error && (
                <div className="toast error">
                    <span>⚠️</span>
                    <span>{error}</span>
                </div>
            )}
        </div>
    )
}

export default App
