# 🛡 Security Policy

## Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability in this project, please report it responsibly.

### How to Report

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please report via one of the following methods:

1. **GitHub Issues** (for non-sensitive issues): [github.com/H0NEYP0T-466/CKD-Ensemble-Classifier/issues](https://github.com/H0NEYP0T-466/CKD-Ensemble-Classifier/issues)
2. **GitHub Security Advisories**: Use the "Report a vulnerability" button on the Security tab

### What to Include

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment:** Within 48 hours
- **Investigation:** Within 7 days
- **Fix & Disclosure:** As soon as reasonably possible

## Security Best Practices

When contributing to this project:

- ❌ **Never** commit secrets, API keys, or credentials
- ✅ **Always** use environment variables for sensitive configuration
- ✅ **Validate** all user inputs at system boundaries
- ✅ **Use** parameterized queries to prevent SQL injection
- ✅ **Sanitize** HTML output to prevent XSS
- ✅ **Enable** CSRF protection on all state-changing endpoints
- ✅ **Implement** rate limiting on API endpoints

## Dependency Security

- Keep all dependencies up to date
- Run `npm audit` for Node.js dependencies
- Run `pip audit` for Python dependencies
- Review security advisories before updating

## Deployment Security

When deploying the backend:

- Use HTTPS in production
- Set strong `SECRET_KEY` values
- Restrict CORS origins to known domains
- Keep the server and OS patched
- Use a reverse proxy (nginx/traefik) in production
