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

                if (monacoRef.current) {
                    const flashDecoration = editorRef.current.createDecorationsCollection([
                        {
                            range: new monacoRef.current.Range(lineNumber, 1, lineNumber, 1000),
                            options: {
                                isWholeLine: true,
                                className: 'highlight-flash',
                                minimap: { color: '#00f0ff', position: 1 }
                            }
                        }
                    ])
                    setTimeout(() => flashDecoration.clear(), 1500)
                }
            }
        }
    }))

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
                            minimap: { color: '#00f0ff', position: 1 }
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
                minimap: { color: h.score > 0.7 ? '#ff2d55' : '#ff9500', position: 1 }
            }
        }))
    }, [highlights])

    const handleEditorMount = (editor, monaco) => {
        editorRef.current = editor
        monacoRef.current = monaco

        // Antigravity neon dark theme
        monaco.editor.defineTheme('antigravity-void', {
            base: 'vs-dark',
            inherit: true,
            rules: [
                { token: 'comment', foreground: '5c5e7a', fontStyle: 'italic' },
                { token: 'keyword', foreground: '00f0ff' },
                { token: 'keyword.control', foreground: 'a855f7' },
                { token: 'string', foreground: 'ff6bcb' },
                { token: 'string.escape', foreground: 'fbbf24' },
                { token: 'number', foreground: 'fbbf24' },
                { token: 'function', foreground: '4d7cff' },
                { token: 'variable', foreground: 'eef0ff' },
                { token: 'type', foreground: '39ff14' },
                { token: 'operator', foreground: '00f0ff' },
                { token: 'decorator', foreground: 'ff9500' },
                { token: 'constant', foreground: 'ff2d55' },
            ],
            colors: {
                'editor.background': '#08071200',
                'editor.foreground': '#eef0ff',
                'editor.lineHighlightBackground': '#ffffff06',
                'editor.lineHighlightBorder': '#ffffff08',
                'editor.selectionBackground': '#4d7cff30',
                'editor.inactiveSelectionBackground': '#4d7cff18',
                'editorCursor.foreground': '#00f0ff',
                'editorLineNumber.foreground': '#3a3c5a',
                'editorLineNumber.activeForeground': '#8b8da8',
                'editorIndentGuide.background': '#ffffff0a',
                'editorIndentGuide.activeBackground': '#ffffff15',
                'editorWhitespace.foreground': '#ffffff08',
                'editor.selectionHighlightBackground': '#4d7cff15',
                'editorBracketMatch.background': '#00f0ff18',
                'editorBracketMatch.border': '#00f0ff40',
                'editorGutter.background': '#08071200',
                'minimap.background': '#0a091800',
                'scrollbar.shadow': '#00000000',
                'scrollbarSlider.background': '#ffffff12',
                'scrollbarSlider.hoverBackground': '#00f0ff25',
                'scrollbarSlider.activeBackground': '#00f0ff35',
            }
        })
        monaco.editor.setTheme('antigravity-void')

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
                fontLigatures: true,
                lineHeight: 24,
                padding: { top: 16, bottom: 16 },
                minimap: { enabled: true, scale: 1, renderCharacters: false },
                scrollBeyondLastLine: false,
                smoothScrolling: true,
                cursorBlinking: 'smooth',
                cursorSmoothCaretAnimation: 'on',
                cursorStyle: 'line',
                cursorWidth: 2,
                renderLineHighlight: 'all',
                automaticLayout: true,
                glyphMargin: true,
                folding: true,
                bracketPairColorization: { enabled: true },
                guides: { bracketPairs: true, indentation: true },
                overviewRulerLanes: 0,
                hideCursorInOverviewRuler: true,
                overviewRulerBorder: false,
            }}
        />
    )
})

export default CodeEditor
