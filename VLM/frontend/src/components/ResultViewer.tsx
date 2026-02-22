import React from 'react';
import './ResultViewer.css';

interface ResultViewerProps {
    data: any;
}

const ResultViewer: React.FC<ResultViewerProps> = ({ data }) => {
    if (!data) return null;

    return (
        <div className="result-viewer">
            <h2>Processing Result</h2>

            <div className="results-grid">
                <div className="result-section">
                    <h3>Extracted Text</h3>
                    <pre className="text-content">{data.text || "No text extracted"}</pre>
                </div>

                {data.html && (
                    <div className="result-section">
                        <h3>HTML Rendition</h3>
                        <div className="html-content" dangerouslySetInnerHTML={{ __html: data.html }} />
                    </div>
                )}
            </div>

            <div className="raw-json">
                <h3>Raw Data</h3>
                <pre>{JSON.stringify(data, null, 2)}</pre>
            </div>
        </div>
    );
};

export default ResultViewer;
