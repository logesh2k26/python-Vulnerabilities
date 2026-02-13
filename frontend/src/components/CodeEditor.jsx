import Editor from '@monaco-editor/react'
import { useMemo } from 'react'

function CodeEditor({ value, onChange, highlights = [] }) {
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
}

export default CodeEditor
