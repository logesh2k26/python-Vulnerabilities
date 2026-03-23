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
                                minimap: { color: '#4c56af', position: 1 }
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
                            minimap: { color: '#4c56af', position: 1 }
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
                minimap: { color: h.score > 0.7 ? '#9f403d' : '#c4501a', position: 1 }
            }
        }))
    }, [highlights])

    const handleEditorMount = (editor, monaco) => {
        editorRef.current = editor
        monacoRef.current = monaco

        // Light theme matching the reference design
        monaco.editor.defineTheme('pyvuln-light', {
            base: 'vs',
            inherit: true,
            rules: [
                { token: 'comment', foreground: 'a8b3bb', fontStyle: 'italic' },
                { token: 'keyword', foreground: '4c56af', fontStyle: 'bold' },
                { token: 'keyword.control', foreground: '4c56af', fontStyle: 'bold' },
                { token: 'string', foreground: '4d626c' },
                { token: 'string.escape', foreground: '9f403d' },
                { token: 'number', foreground: 'c4501a' },
                { token: 'function', foreground: '4049a2' },
                { token: 'variable', foreground: '29343a' },
                { token: 'type', foreground: '415368' },
                { token: 'operator', foreground: '4c56af' },
                { token: 'decorator', foreground: '4f6176' },
                { token: 'constant', foreground: '9f403d' },
                { token: 'delimiter', foreground: '566168' },
            ],
            colors: {
                'editor.background': '#fbfcfd',
                'editor.foreground': '#29343a',
                'editor.lineHighlightBackground': '#f0f4f808',
                'editor.lineHighlightBorder': '#e8eff410',
                'editor.selectionBackground': '#e0e0ff50',
                'editor.inactiveSelectionBackground': '#e0e0ff30',
                'editorCursor.foreground': '#4c56af',
                'editorLineNumber.foreground': '#a8b3bb',
                'editorLineNumber.activeForeground': '#566168',
                'editorIndentGuide.background': '#e8eff4',
                'editorIndentGuide.activeBackground': '#d9e4ec',
                'editorWhitespace.foreground': '#e8eff410',
                'editor.selectionHighlightBackground': '#e0e0ff25',
                'editorBracketMatch.background': '#e0e0ff40',
                'editorBracketMatch.border': '#4c56af40',
                'editorGutter.background': '#fbfcfd',
                'minimap.background': '#f7f9fc',
                'scrollbar.shadow': '#00000008',
                'scrollbarSlider.background': '#a8b3bb20',
                'scrollbarSlider.hoverBackground': '#a8b3bb40',
                'scrollbarSlider.activeBackground': '#a8b3bb50',
            }
        })
        monaco.editor.setTheme('pyvuln-light')

        if (decorations.length > 0) {
            editor.createDecorationsCollection(decorations)
        }
    }

    return (
        <>
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
            <style>{`
                /* Light theme highlight styles */
                .highlight-critical {
                    background: rgba(159, 64, 61, 0.08) !important;
                    border-left: 4px solid #9f403d !important;
                }
                .highlight-high {
                    background: rgba(196, 80, 26, 0.06) !important;
                    border-left: 4px solid #c4501a !important;
                }
                .highlight-medium {
                    background: rgba(76, 86, 175, 0.06) !important;
                    border-left: 4px solid #4c56af !important;
                }
                .highlight-flash {
                    background: rgba(76, 86, 175, 0.12) !important;
                    border-left: 4px solid #4c56af !important;
                    transition: background 0.3s ease;
                }
                .glyph-vulnerability {
                    background: #9f403d;
                    border-radius: 50%;
                    margin-left: 4px;
                    width: 8px !important;
                    height: 8px !important;
                }
            `}</style>
        </>
    )
})

export default CodeEditor
