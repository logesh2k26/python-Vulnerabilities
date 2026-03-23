# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. Email: **[logesh2k26@gmail.com]** with the subject line `[SECURITY] PyVulnDetect Vulnerability Report`
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix & Disclosure**: Coordinated with reporter

## Security Best Practices

### Environment Variables
- **Never** commit `.env` files to version control
- Use `.env.example` as a template for required variables
- Rotate API keys if you suspect they have been exposed

### API Keys
- All API keys must be stored in environment variables
- The backend validates API keys via the `X-API-Key` header
- Frontend communicates with backend endpoints only — no direct third-party API calls with keys

### Dependencies
- Regularly update Python and Node.js dependencies
- Run `pip audit` and `npm audit` periodically
