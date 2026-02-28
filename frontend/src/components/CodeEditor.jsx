import Editor from '@monaco-editor/react'
import { useMemo, useRef, useImperativeHandle, forwardRef, useEffect } from 'react'

const CodeEditor = forwardRef(function CodeEditor({ value, onChange, highlights = [], targetLine }, ref) {
    const editorRef = useRef(null)
    const monacoRef = useRef(null)

    useImperativeHandle(ref, () => ({
        scrollToLine(lineNumber) {
            if (editorRef.current) {
                editorRef.current.revealLineInCenter(lineNumber)
                editorRef.current.setPosition({ lineNumber, column: 1 })
                editorRef.current.focus()

                // Flash highlight on the target line
                if (monacoRef.current) {
                    const flashDecoration = editorRef.current.createDecorationsCollection([
                        {
                            range: new monacoRef.current.Range(lineNumber, 1, lineNumber, 1000),
                            options: {
                                isWholeLine: true,
                                className: 'highlight-flash',
                                minimap: { color: '#00d4ff', position: 1 }
                            }
                        }
                    ])
                    // Remove flash after animation
                    setTimeout(() => flashDecoration.clear(), 1500)
                }
            }
        }
    }))

    // React to targetLine changes from parent
    useEffect(() => {
        if (targetLine && editorRef.current) {
            editorRef.current.revealLineInCenter(targetLine)
            editorRef.current.setPosition({ lineNumber: targetLine, column: 1 })
            editorRef.current.focus()

            if (monacoRef.current) {
                const flashDecoration = editorRef.current.createDecorationsCollection([
                    {
                        range: new monacoRef.current.Range(targetLine, 1, targetLine, 1000),
                        options: {
                            isWholeLine: true,
                            className: 'highlight-flash',
                            minimap: { color: '#00d4ff', position: 1 }
                        }
                    }
                ])
                setTimeout(() => flashDecoration.clear(), 1500)
            }
        }
    }, [targetLine])

    const decorations = useMemo(() => {
        return highlights.map(h => ({
            range: {
                startLineNumber: h.line,
                startColumn: 1,
                endLineNumber: h.line,
                endColumn: 1000
            },
            options: {
                isWholeLine: true,
                className: h.score > 0.7 ? 'highlight-critical' :
                    h.score > 0.4 ? 'highlight-high' : 'highlight-medium',
                glyphMarginClassName: 'glyph-vulnerability',
                minimap: { color: h.score > 0.7 ? '#ef4444' : '#eab308', position: 1 }
            }
        }))
    }, [highlights])

    const handleEditorMount = (editor, monaco) => {
        editorRef.current = editor
        monacoRef.current = monaco

        monaco.editor.defineTheme('vuln-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
                { token: 'comment', foreground: '6b7280', fontStyle: 'italic' },
                { token: 'keyword', foreground: 'c084fc' },
                { token: 'string', foreground: '34d399' },
                { token: 'number', foreground: 'fbbf24' },
                { token: 'function', foreground: '60a5fa' },
            ],
            colors: {
                'editor.background': '#12121a',
                'editor.foreground': '#f8fafc',
                'editor.lineHighlightBackground': '#1e1e2e',
                'editor.selectionBackground': '#3b82f680',
                'editorCursor.foreground': '#00d4ff',
                'editorLineNumber.foreground': '#4b5563',
                'editorLineNumber.activeForeground': '#9ca3af',
                'editor.inactiveSelectionBackground': '#3b82f640',
            }
        })
        monaco.editor.setTheme('vuln-dark')

        if (decorations.length > 0) {
            editor.createDecorationsCollection(decorations)
        }
    }

    return (
        <Editor
            height="100%"
            language="python"
            value={value}
            onChange={onChange}
            onMount={handleEditorMount}
            options={{
                fontSize: 14,
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                lineHeight: 24,
                padding: { top: 16, bottom: 16 },
                minimap: { enabled: true, scale: 1 },
                scrollBeyondLastLine: false,
                smoothScrolling: true,
                cursorBlinking: 'smooth',
                cursorSmoothCaretAnimation: 'on',
                renderLineHighlight: 'all',
                automaticLayout: true,
                glyphMargin: true,
                folding: true,
                bracketPairColorization: { enabled: true }
            }}
        />
    )
})

export default CodeEditor
