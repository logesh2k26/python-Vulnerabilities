import { useMemo } from 'react'

function TraceVisualizer({ taintPath }) {
    if (!taintPath) return null

    const steps = useMemo(() => {
        const list = []
        // Step 1: Source
        list.push({
            type: 'source',
            line: taintPath.source_line,
            desc: `Source: ${taintPath.source_variable} (${taintPath.source_type})`,
            icon: '🛑'
        })

        // Step 2: Propagation
        if (taintPath.propagation_path) {
            taintPath.propagation_path.forEach(step => {
                list.push({
                    type: 'flow',
                    line: '?', // Node ID is distinct from line, description might have it
                    desc: step.description,
                    icon: '⬇️'
                })
            })
        }

        // Step 3: Sink
        list.push({
            type: 'sink',
            line: taintPath.sink_line,
            desc: `Sink: ${taintPath.sink_function} (${taintPath.sink_type})`,
            icon: '💥'
        })
        return list
    }, [taintPath])

    return (
        <div className="trace-visualizer">
            <h4 className="trace-title">🛡️ Attack Trace</h4>
            <div className="trace-steps">
                {steps.map((step, idx) => (
                    <div key={idx} className={`trace-step step-${step.type}`}>
                        <div className="step-icon">{step.icon}</div>
                        <div className="step-content">
                            <span className="step-desc">{step.desc}</span>
                            {step.line && step.line !== '?' && (
                                <span className="step-line">Line {step.line}</span>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            <style jsx>{`
                .trace-visualizer {
                    margin-top: 1rem;
                    padding: 1rem;
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                    border-left: 3px solid #ef4444;
                }
                .trace-title {
                    margin: 0 0 0.5rem 0;
                    color: #f87171;
                    font-size: 0.9rem;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                .trace-steps {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }
                .trace-step {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    padding: 0.5rem;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 4px;
                    font-size: 0.85rem;
                }
                .step-source { border-left: 2px solid #ef4444; }
                .step-flow { border-left: 2px solid #f59e0b; margin-left: 1rem; }
                .step-sink { border-left: 2px solid #ef4444; }
                .step-line {
                    margin-left: auto;
                    font-family: monospace;
                    background: rgba(0,0,0,0.3);
                    padding: 2px 6px;
                    border-radius: 4px;
                    color: #9ca3af;
                }
            `}</style>
        </div>
    )
}

export default TraceVisualizer
