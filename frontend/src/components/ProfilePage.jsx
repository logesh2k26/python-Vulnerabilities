import React, { useState } from 'react'

function ProfilePage({ user, onLogout }) {
    const displayUser = user || { 
        name: 'Alex Rivera', 
        email: 'alex.rivera@skydrop.io', 
        role: 'Senior Security Architect', 
        scans: '1,284', 
        joined: "Mar '22" 
    }


    return (
        <div className="ps-container">
            {/* Page Header */}
            <header className="ps-header">
                <h1 className="ps-title">Profile Info</h1>
                <p className="ps-subtitle">Your identity and creator credentials.</p>
            </header>

            <div className="ps-layout">
                {/* Left Column: Stats & Health */}
                <div className="ps-side-col">
                    <div className="ps-card ps-profile-card">
                        <div className="ps-avatar-wrap">
                            <img 
                                src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${displayUser.name}`} 
                                alt="Avatar" 
                                className="ps-avatar-img"
                            />
                            <button className="ps-avatar-edit">
                                <span className="material-symbols-outlined">edit</span>
                            </button>
                        </div>
                        
                        <span className="ps-badge">VERIFIED EXPERT</span>
                        
                        <h2 className="ps-display-name">{displayUser.name}</h2>
                        <p className="ps-display-role">{displayUser.role}</p>

                        <div className="ps-stats-grid">
                            <div className="ps-stat-box">
                                <span className="ps-stat-label">TOTAL SCANS</span>
                                <span className="ps-stat-value">{displayUser.scans}</span>
                            </div>
                            <div className="ps-stat-box">
                                <span className="ps-stat-label">JOINED DATE</span>
                                <span className="ps-stat-value">{displayUser.joined}</span>
                            </div>
                        </div>
                    </div>

                    <div className="ps-card ps-health-card">
                        <div className="ps-health-icon">
                            <span className="material-symbols-outlined">shield</span>
                        </div>
                        <div className="ps-health-info">
                            <h4 className="ps-health-title">Account Health</h4>
                            <p className="ps-health-desc">Last audited 2 days ago</p>
                        </div>
                        <div className="ps-health-value">98%</div>
                    </div>
                </div>

                {/* Right Column: Account Details */}
                <div className="ps-main-col">
                    <div className="ps-card ps-details-card">
                        <div className="ps-details-header">
                            <div className="ps-details-title-wrap">
                                <span className="material-symbols-outlined ps-details-icon">badge</span>
                                <h3 className="ps-details-title">Account Details</h3>
                            </div>
                            <button className="ps-reset-btn">Reset Defaults</button>
                        </div>

                        <div className="ps-form">
                            <div className="ps-form-grid">
                                <div className="ps-field">
                                    <label className="ps-label">DISPLAY NAME</label>
                                    <input type="text" className="ps-input" defaultValue={displayUser.name} />
                                </div>


                            </div>

                            <div className="ps-field ps-field-full">
                                <label className="ps-label">EMAIL ADDRESS</label>
                                <div className="ps-input-with-icon">
                                    <span className="material-symbols-outlined ps-field-icon">mail</span>
                                    <input type="email" className="ps-input" defaultValue={displayUser.email} />
                                </div>
                            </div>



                            <div className="ps-form-actions">
                                <button className="ps-save-btn">Save Changes</button>
                                <button className="ps-cancel-btn">Cancel</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Creator Info Section */}
            <div className="ps-creator-section">
                <div className="ps-card ps-creator-card">
                    <div className="ps-details-header">
                        <div className="ps-details-title-wrap">
                            <span className="material-symbols-outlined ps-details-icon">person</span>
                            <h3 className="ps-details-title">Creator Info</h3>
                        </div>
                    </div>
                    <div className="ps-creator-content">
                        <div className="ps-creator-avatar">
                            <div className="ps-creator-img">
                                <span className="ps-creator-initial">L</span>
                            </div>
                        </div>
                        <div className="ps-creator-details">
                            <span className="ps-creator-label">DESIGNED & BUILT BY</span>
                            <h2 className="ps-creator-name">Logesh D</h2>
                            <p className="ps-creator-quote">
                                "We aren't just building software; we are crafting the digital infrastructure that keeps our collective data secure. Precision is not optional."
                            </p>
                            <div className="ps-creator-links">
                                <a href="https://github.com/logesh2k26" target="_blank" rel="noopener noreferrer" className="ps-creator-link">
                                    <span className="material-symbols-outlined">code</span>
                                    <span>View GitHub</span>
                                </a>
                                <a href="https://www.linkedin.com/in/logesh-d-b9ba442aa/" target="_blank" rel="noopener noreferrer" className="ps-creator-link">
                                    <span className="material-symbols-outlined">person_search</span>
                                    <span>Connect on LinkedIn</span>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <footer className="ps-footer">
                <div className="ps-footer-left">
                    <span className="material-symbols-outlined">history</span>
                    <span>Last login: Oct 24, 2023 14:32 GMT</span>
                    <span className="ps-footer-sep">•</span>
                    <span className="material-symbols-outlined">public</span>
                    <span>IP: 192.168.1.104</span>
                </div>
                <button className="ps-deactivate-btn">
                    <span className="material-symbols-outlined">logout</span>
                    Deactivate Account
                </button>
            </footer>

            <style>{`
                .ps-container {
                    padding: 40px 48px;
                    max-width: 1200px;
                    margin: 0 auto;
                }

                /* Header */
                .ps-header { margin-bottom: 40px; }
                .ps-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 32px;
                    font-weight: 800;
                    color: #0f172a;
                    margin: 0 0 8px;
                }
                .ps-subtitle {
                    font-size: 15px;
                    color: #64748b;
                    margin: 0;
                }

                /* Layout */
                .ps-layout {
                    display: grid;
                    grid-template-columns: 340px 1fr;
                    gap: 32px;
                    margin-bottom: 48px;
                }

                /* Cards */
                .ps-card {
                    background: #fff;
                    border-radius: 12px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                }

                /* Profile Card */
                .ps-profile-card {
                    padding: 40px 24px;
                    text-align: center;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                .ps-avatar-wrap {
                    position: relative;
                    margin-bottom: 24px;
                }
                .ps-avatar-img {
                    width: 120px;
                    height: 120px;
                    border-radius: 20px;
                    background: #ffedd5;
                }
                .ps-avatar-edit {
                    position: absolute;
                    bottom: -8px;
                    right: -8px;
                    width: 32px;
                    height: 32px;
                    background: #4f46e5;
                    border: 3px solid #fff;
                    border-radius: 50%;
                    color: #fff;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                }
                .ps-avatar-edit span { font-size: 16px; }

                .ps-badge {
                    font-size: 10px;
                    font-weight: 800;
                    letter-spacing: 0.1em;
                    color: #4f46e5;
                    padding: 4px 10px;
                    background: #eef2ff;
                    border-radius: 6px;
                    margin-bottom: 16px;
                }
                .ps-display-name {
                    font-family: 'Manrope', sans-serif;
                    font-size: 24px;
                    font-weight: 800;
                    color: #0f172a;
                    margin: 0 0 4px;
                }
                .ps-display-role {
                    font-size: 14px;
                    font-weight: 600;
                    color: #4f46e5;
                    margin-bottom: 32px;
                }

                .ps-stats-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 12px;
                    width: 100%;
                }
                .ps-stat-box {
                    background: #f8fafc;
                    padding: 16px;
                    border-radius: 10px;
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                .ps-stat-label {
                    font-size: 9px;
                    font-weight: 800;
                    color: #94a3b8;
                    letter-spacing: 0.05em;
                }
                .ps-stat-value {
                    font-family: 'Manrope', sans-serif;
                    font-size: 18px;
                    font-weight: 800;
                    color: #1e293b;
                }

                /* Health Card */
                .ps-health-card {
                    margin-top: 16px;
                    padding: 20px 24px;
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    background: #f1f5f9;
                }
                .ps-health-icon {
                    width: 40px;
                    height: 40px;
                    background: #fff;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #4f46e5;
                }
                .ps-health-info { flex: 1; }
                .ps-health-title {
                    font-size: 14px;
                    font-weight: 700;
                    color: #1e293b;
                    margin: 0;
                }
                .ps-health-desc {
                    font-size: 12px;
                    color: #64748b;
                    margin: 2px 0 0;
                }
                .ps-health-value {
                    font-family: 'Manrope', sans-serif;
                    font-size: 16px;
                    font-weight: 800;
                    color: #4f46e5;
                }

                /* Details Card */
                .ps-details-card {
                    padding: 32px;
                    background: #f8fafc;
                    min-height: 100%;
                }
                .ps-details-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 32px;
                }
                .ps-details-title-wrap {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                .ps-details-icon { color: #4f46e5; }
                .ps-details-title {
                    font-family: 'Manrope', sans-serif;
                    font-size: 18px;
                    font-weight: 800;
                    color: #1e293b;
                    margin: 0;
                }
                .ps-reset-btn {
                    background: none;
                    border: none;
                    color: #4f46e5;
                    font-size: 13px;
                    font-weight: 700;
                    cursor: pointer;
                }

                /* Form Components */
                .ps-form { display: flex; flex-direction: column; gap: 24px; }
                .ps-form-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                .ps-field { display: flex; flex-direction: column; gap: 8px; }
                .ps-label {
                    font-size: 11px;
                    font-weight: 800;
                    color: #64748b;
                    letter-spacing: 0.05em;
                }
                .ps-input {
                    background: #fff;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 12px 16px;
                    font-size: 14px;
                    color: #0f172a;
                    outline: none;
                }
                .ps-input:focus { border-color: #4f46e5; }
                
                .ps-input-with-icon { position: relative; display: flex; align-items: center; }
                .ps-field-icon {
                    position: absolute;
                    left: 14px;
                    font-size: 18px;
                    color: #94a3b8;
                }
                .ps-input-with-icon .ps-input { padding-left: 44px; width: 100%; }

                /* API Section */
                .ps-api-section {
                    background: #fff;
                    border-radius: 12px;
                    padding: 24px;
                    margin-top: 8px;
                }
                .ps-api-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .ps-api-title {
                    font-size: 14px;
                    font-weight: 700;
                    color: #1e293b;
                    margin: 0;
                }
                .ps-api-desc {
                    font-size: 12px;
                    color: #64748b;
                    margin: 4px 0 0;
                }
                
                .ps-api-key-wrap {
                    background: #f1f5f9;
                    padding: 12px 16px;
                    border-radius: 8px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .ps-api-key {
                    font-family: var(--font-code);
                    font-size: 13px;
                    color: #4f46e5;
                }
                .ps-copy-btn {
                    background: none;
                    border: none;
                    color: #94a3b8;
                    cursor: pointer;
                }
                .ps-copy-btn span { font-size: 18px; }

                /* Toggle Switch */
                .ps-switch { position: relative; width: 44px; height: 24px; cursor: pointer; }
                .ps-switch input { opacity: 0; width: 0; height: 0; }
                .ps-switch-slider {
                    position: absolute;
                    inset: 0;
                    background: #cbd5e1;
                    border-radius: 20px;
                    transition: .3s;
                }
                .ps-switch-slider:before {
                    position: absolute;
                    content: "";
                    height: 18px;
                    width: 18px;
                    left: 3px;
                    bottom: 3px;
                    background: white;
                    border-radius: 50%;
                    transition: .3s;
                }
                .ps-switch input:checked + .ps-switch-slider { background: #4f46e5; }
                .ps-switch input:checked + .ps-switch-slider:before { transform: translateX(20px); }

                /* Actions */
                .ps-form-actions {
                    display: flex;
                    gap: 16px;
                    margin-top: 12px;
                }
                .ps-save-btn {
                    padding: 12px 32px;
                    background: #4f46e5;
                    color: #fff;
                    border: none;
                    border-radius: 8px;
                    font-weight: 700;
                    font-size: 14px;
                    cursor: pointer;
                }
                .ps-cancel-btn {
                    padding: 12px 24px;
                    background: none;
                    border: none;
                    color: #64748b;
                    font-weight: 600;
                    font-size: 14px;
                    cursor: pointer;
                }

                /* Footer */
                .ps-footer {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding-top: 32px;
                }
                .ps-footer-left {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 12px;
                    color: #64748b;
                    font-weight: 500;
                }
                .ps-footer-left span.material-symbols-outlined { font-size: 16px; }
                .ps-footer-sep { margin: 0 4px; opacity: 0.5; }

                .ps-deactivate-btn {
                    background: none;
                    border: none;
                    color: #b91c1c;
                    font-size: 13px;
                    font-weight: 700;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    cursor: pointer;
                }
                .ps-deactivate-btn span { font-size: 18px; }

                @media (max-width: 900px) {
                    .ps-layout { grid-template-columns: 1fr; }
                    .ps-form-grid { grid-template-columns: 1fr; }
                }

                /* Creator Info Section */
                .ps-creator-section {
                    margin-top: 32px;
                    max-width: 1200px;
                }
                .ps-creator-card {
                    padding: 32px;
                }
                .ps-creator-content {
                    display: flex;
                    gap: 32px;
                    align-items: flex-start;
                    margin-top: 24px;
                }
                .ps-creator-avatar {
                    flex-shrink: 0;
                }
                .ps-creator-img {
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #4f46e5, #7c3aed);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
                }
                .ps-creator-initial {
                    font-size: 32px;
                    font-weight: 800;
                    color: #fff;
                    font-family: 'Manrope', sans-serif;
                }
                .ps-creator-details {
                    flex: 1;
                }
                .ps-creator-label {
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 1.5px;
                    color: #94a3b8;
                    text-transform: uppercase;
                }
                .ps-creator-name {
                    font-family: 'Manrope', sans-serif;
                    font-size: 28px;
                    font-weight: 800;
                    color: #0f172a;
                    margin: 6px 0 12px;
                }
                .ps-creator-quote {
                    font-size: 14px;
                    color: #64748b;
                    line-height: 1.7;
                    font-style: italic;
                    margin: 0 0 20px;
                    border-left: 3px solid #e2e8f0;
                    padding-left: 16px;
                }
                .ps-creator-links {
                    display: flex;
                    gap: 12px;
                }
                .ps-creator-link {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    padding: 10px 20px;
                    background: #f1f5f9;
                    border-radius: 8px;
                    color: #334155;
                    text-decoration: none;
                    font-size: 13px;
                    font-weight: 600;
                    transition: all 0.2s;
                }
                .ps-creator-link:hover {
                    background: #4f46e5;
                    color: #fff;
                }
                .ps-creator-link span.material-symbols-outlined {
                    font-size: 18px;
                }
                @media (max-width: 600px) {
                    .ps-creator-content { flex-direction: column; align-items: center; text-align: center; }
                    .ps-creator-quote { border-left: none; padding-left: 0; }
                    .ps-creator-links { justify-content: center; }
                }
            `}</style>
        </div>
    )
}

export default ProfilePage
