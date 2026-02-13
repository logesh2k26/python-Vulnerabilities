import { useState, useCallback } from 'react'

function FileUpload({ onUpload }) {
    const [isDragging, setIsDragging] = useState(false)

    const handleDrag = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
    }, [])

    const handleDragIn = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(true)
    }, [])

    const handleDragOut = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)

        const files = [...e.dataTransfer.files].filter(f => f.name.endsWith('.py'))
        if (files.length > 0) {
            onUpload(files)
        }
    }, [onUpload])

    const handleFileSelect = (e) => {
        const files = [...e.target.files].filter(f => f.name.endsWith('.py'))
        if (files.length > 0) {
            onUpload(files)
        }
    }

    return (
        <div
            className={`upload-zone ${isDragging ? 'drag-over' : ''}`}
            onDragEnter={handleDragIn}
            onDragLeave={handleDragOut}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            style={{ margin: '0 24px 24px' }}
        >
            <input
                type="file"
                id="file-input"
                accept=".py"
                multiple
                onChange={handleFileSelect}
                style={{ display: 'none' }}
            />
            <label htmlFor="file-input" style={{ cursor: 'pointer', display: 'block' }}>
                <div className="upload-icon">📂</div>
                <h3 className="upload-title">Drop Python Files Here</h3>
                <p className="upload-subtitle">or click to browse • .py files only</p>
            </label>
        </div>
    )
}

export default FileUpload
