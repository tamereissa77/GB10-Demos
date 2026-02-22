import React, { useState, useEffect } from 'react';

import './FileUpload.css';

interface FileUploadProps {
    onUploadSuccess: (data: any) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess }) => {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const [backendReady, setBackendReady] = useState(false);
    const [backendStatus, setBackendStatus] = useState<string>('Checking backend...');

    // Poll /api/health until the backend model is ready
    useEffect(() => {
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout>;

        const checkHealth = async () => {
            try {
                const resp = await fetch('/api/health');
                const data = await resp.json();
                if (cancelled) return;

                if (data.ready) {
                    setBackendReady(true);
                    setBackendStatus('Backend ready ✅');
                    return; // stop polling
                } else {
                    setBackendStatus(data.message || 'OCR model is loading...');
                }
            } catch (_e) {
                if (cancelled) return;
                setBackendStatus('Waiting for backend to start...');
            }
            // Poll again in 3 seconds
            timer = setTimeout(checkHealth, 3000);
        };

        checkHealth();

        return () => {
            cancelled = true;
            clearTimeout(timer);
        };
    }, []);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0]);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);
        setProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const contentType = response.headers.get('content-type') || '';
                const bodyText = await response.text();
                throw new Error(
                    `Upload failed: HTTP ${response.status}. ` +
                    `Content-Type: ${contentType}. ` +
                    `Body: ${bodyText.slice(0, 200)}`
                );
            }

            if (!response.body) throw new Error("No response body");
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulatedText = "";
            let accumulatedHtml = "";
            let processedPages = 0;
            // Buffer for incomplete NDJSON lines that span multiple network chunks
            let lineBuffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                // Prepend any leftover partial line from the previous chunk
                lineBuffer += chunk;

                // Split on newlines; the last element may be an incomplete line
                const parts = lineBuffer.split('\n');
                // Keep the last (potentially incomplete) part in the buffer
                lineBuffer = parts.pop() || "";

                for (const line of parts) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;

                    let data: any;
                    try {
                        data = JSON.parse(trimmed);
                    } catch (_e) {
                        // Malformed line – skip silently
                        continue;
                    }

                    if (data.type === 'progress') {
                        processedPages = data.page;
                        const total = data.total;
                        const pct = Math.round((processedPages / total) * 100);
                        setProgress(pct);

                        // Accumulate page content (skip progress-only messages with empty page_data)
                        if (data.page_data && (data.page_data.markdown || data.page_data.html)) {
                            accumulatedText += (accumulatedText ? "\n\n" : "") + (data.page_data.markdown || "");
                            accumulatedHtml += (accumulatedHtml ? "\n<hr>\n" : "") + (data.page_data.html || "");
                        }

                        // Update the parent component with partial results
                        onUploadSuccess({
                            text: accumulatedText,
                            html: accumulatedHtml,
                            raw_count: processedPages,
                            status: "processing"
                        });
                    } else if (data.type === 'complete') {
                        onUploadSuccess(data);
                    } else if (data.type === 'error') {
                        throw new Error(data.error);
                    }
                }
            }

            // Process any remaining data left in the buffer after the stream ends
            if (lineBuffer.trim()) {
                try {
                    const data = JSON.parse(lineBuffer.trim());
                    if (data.type === 'complete') {
                        onUploadSuccess(data);
                    } else if (data.type === 'error') {
                        throw new Error(data.error);
                    }
                } catch (_e) {
                    // Final fragment wasn't valid JSON – ignore
                }
            }
        } catch (err: any) {
            setError(err.message || 'Upload failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="file-upload-container">
            <h3>Upload Arabic Document</h3>

            {/* Backend readiness indicator */}
            <div className={`backend-status ${backendReady ? 'ready' : 'loading'}`}>
                {backendReady ? '🟢' : '🟡'} {backendStatus}
            </div>

            <div className="upload-box">
                <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={handleFileChange} />
                {file && (
                    <button
                        onClick={handleUpload}
                        disabled={loading || !backendReady}
                        className="upload-btn"
                        title={!backendReady ? 'Waiting for OCR model to load...' : ''}
                    >
                        {loading ? `Processing ${progress}%` : !backendReady ? 'Model Loading...' : 'Process Document'}
                    </button>
                )}
            </div>
            {error && <p className="error-message">{error}</p>}
        </div>
    );
};

export default FileUpload;
