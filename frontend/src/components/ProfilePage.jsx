import React, { useState } from 'react'

function ProfilePage({ user, onLogout }) {
    // If we have a user prop, show profile directly
    const displayUser = user || { name: 'User', email: 'user@example.com', role: 'Security Analyst', scans: 0, joined: '2026-01-01' }

    return (
        <div className="page-container">
            <div className="page-header">
                <h1 className="page-title">👤 Profile</h1>
                <p className="page-subtitle">Manage your account</p>
            </div>

            <div className="profile-layout">
                <div className="glass-card profile-card">
                    <div className="profile-avatar-large">
                        {displayUser.name.charAt(0).toUpperCase()}
                    </div>
                    <h2 className="profile-name">{displayUser.name}</h2>
                    <p className="profile-email">{displayUser.email}</p>
                    <span className="profile-role">{displayUser.role}</span>

                    <div className="profile-stats-row">
                        <div className="profile-stat">
                            <span className="profile-stat-value neon-text-cyan">{displayUser.scans}</span>
                            <span className="profile-stat-label">Scans</span>
                        </div>
                        <div className="profile-stat-divider" />
                        <div className="profile-stat">
                            <span className="profile-stat-value neon-text-green">Joined</span>
                            <span className="profile-stat-label">{displayUser.joined}</span>
                        </div>
                    </div>

                    {onLogout && (
                        <button className="profile-logout-btn" onClick={onLogout}>
                            🚪 Sign Out
                        </button>
                    )}
                </div>

                <div className="glass-card profile-details">
                    <h3 className="profile-section-title">Account Details</h3>
                    <div className="profile-detail-row">
                        <span className="profile-detail-label">Display Name</span>
                        <span className="profile-detail-value">{displayUser.name}</span>
                    </div>
                    <div className="profile-detail-row">
                        <span className="profile-detail-label">Email</span>
                        <span className="profile-detail-value">{displayUser.email}</span>
                    </div>
                    <div className="profile-detail-row">
                        <span className="profile-detail-label">Role</span>
                        <span className="profile-detail-value">{displayUser.role}</span>
                    </div>
                    <div className="profile-detail-row">
                        <span className="profile-detail-label">API Access</span>
                        <span className="profile-detail-value" style={{ color: 'var(--neon-green)' }}>✓ Active</span>
                    </div>
                </div>
            </div>

            <ProfileStyles />
        </div>
    )
}

function ProfileStyles() {
    return (
        <style>{`
            .profile-layout {
                display: grid;
                grid-template-columns: 320px 1fr;
                gap: 24px;
                max-width: 900px;
            }
            @media (max-width: 800px) {
                .profile-layout { grid-template-columns: 1fr; }
            }
            .profile-card {
                padding: 36px 28px;
                text-align: center;
            }
            .profile-avatar-large {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--neon-cyan), var(--neon-violet));
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: var(--font-heading);
                font-size: 32px;
                font-weight: 700;
                color: white;
                margin: 0 auto 16px;
                box-shadow: 0 0 30px rgba(0, 240, 255, 0.15);
            }
            .profile-name {
                font-family: var(--font-heading);
                font-size: 20px;
                font-weight: 700;
                color: var(--text-primary);
                margin: 0 0 4px;
            }
            .profile-email {
                font-size: 13px;
                color: var(--text-muted);
                margin: 0 0 12px;
            }
            .profile-role {
                display: inline-block;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 12px;
                border-radius: 8px;
                background: rgba(0, 240, 255, 0.06);
                color: var(--neon-cyan);
                border: 1px solid rgba(0, 240, 255, 0.12);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 20px;
            }
            .profile-stats-row {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 24px;
                padding: 18px 0;
                margin-bottom: 20px;
                border-top: 1px solid rgba(120, 100, 255, 0.06);
                border-bottom: 1px solid rgba(120, 100, 255, 0.06);
            }
            .profile-stat {
                display: flex;
                flex-direction: column;
                gap: 2px;
            }
            .profile-stat-value {
                font-family: var(--font-heading);
                font-size: 18px;
                font-weight: 700;
            }
            .profile-stat-label {
                font-size: 11px;
                color: var(--text-muted);
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }
            .profile-stat-divider {
                width: 1px;
                height: 36px;
                background: rgba(120, 100, 255, 0.1);
            }
            .profile-logout-btn {
                width: 100%;
                padding: 12px;
                border: 1px solid rgba(255, 45, 85, 0.15);
                border-radius: 12px;
                background: rgba(255, 45, 85, 0.06);
                color: var(--neon-red);
                font-family: var(--font-main);
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.25s;
            }
            .profile-logout-btn:hover {
                background: rgba(255, 45, 85, 0.12);
                box-shadow: 0 0 16px rgba(255, 45, 85, 0.1);
            }
            .profile-details {
                padding: 28px;
            }
            .profile-section-title {
                font-family: var(--font-heading);
                font-size: 15px;
                font-weight: 600;
                color: var(--text-primary);
                margin: 0 0 18px;
                padding-bottom: 12px;
                border-bottom: 1px solid rgba(120, 100, 255, 0.08);
            }
            .profile-detail-row {
                display: flex;
                justify-content: space-between;
                padding: 12px 0;
                border-bottom: 1px solid rgba(120, 100, 255, 0.04);
            }
            .profile-detail-row:last-child { border-bottom: none; }
            .profile-detail-label {
                font-size: 14px;
                color: var(--text-secondary);
            }
            .profile-detail-value {
                font-size: 14px;
                font-weight: 500;
                color: var(--text-primary);
            }
        `}</style>
    )
}

export default ProfilePage
