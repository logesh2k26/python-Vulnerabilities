import { useMemo } from 'react'

function TraceVisualizer({ taintPath }) {
    if (!taintPath) return null

    const steps = useMemo(() => {
        const list = []
        list.push({
            type: 'source',
            line: taintPath.source_line,
            desc: `Source: ${taintPath.source_variable} (${taintPath.source_type})`,
            icon: '🛑'
        })

        if (taintPath.propagation_path) {
            taintPath.propagation_path.forEach(step => {
                list.push({
                    type: 'flow',
                    line: '?',
                    desc: step.description,
                    icon: '⬇️'
                })
            })
        }

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

            <style>{`
                .trace-visualizer {
                    margin-top: 1rem;
                    padding: 14px;
                    background: rgba(255, 45, 85, 0.04);
                    border-radius: 12px;
                    border-left: 3px solid var(--neon-red, #ff2d55);
                    border: 1px solid rgba(255, 45, 85, 0.1);
                }
                .trace-title {
                    margin: 0 0 0.6rem 0;
                    color: var(--neon-red, #ff2d55);
                    font-size: 0.8rem;
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                    font-family: var(--font-heading, 'Outfit');
                    font-weight: 700;
                }
                .trace-steps {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                .trace-step {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    padding: 8px 10px;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                    font-size: 0.82rem;
                    color: var(--text-secondary, #8b8da8);
                    border: 1px solid rgba(120, 100, 255, 0.05);
                    transition: all 0.2s;
                }
                .trace-step:hover {
                    background: rgba(255, 255, 255, 0.06);
                    border-color: rgba(120, 100, 255, 0.12);
                }
                .step-source { border-left: 2px solid var(--neon-red, #ff2d55); }
                .step-flow { border-left: 2px solid var(--neon-orange, #ff9500); margin-left: 1rem; }
                .step-sink { border-left: 2px solid var(--neon-red, #ff2d55); }
                .step-content {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    flex: 1;
                    gap: 8px;
                }
                .step-line {
                    font-family: var(--font-code, monospace);
                    background: rgba(0, 240, 255, 0.06);
                    padding: 2px 8px;
                    border-radius: 6px;
                    color: var(--neon-cyan, #00f0ff);
                    font-size: 0.75rem;
                    border: 1px solid rgba(0, 240, 255, 0.1);
                    white-space: nowrap;
                }
                .step-icon {
                    font-size: 14px;
                    flex-shrink: 0;
                }
            `}</style>
        </div>
    )
}

export default TraceVisualizer
