# Security Policy

## Supported Versions

We take security seriously at Zapomni. This table shows which versions of the project are currently supported with security updates:

| Version | Supported          | Status      | Release Date |
| ------- | ------------------ | ----------- | ------------ |
| 0.1.x   | :white_check_mark: | Active      | 2025-11-24   |
| < 0.1   | :x:                | Development | -            |

Current stable release: **v0.1.0** (Initial Public Release with 18 MCP tools and 2019 comprehensive tests)

## Local-First Security Model

Zapomni is designed with a **local-first architecture** that provides inherent security benefits:

- **No external API dependencies** - All processing happens on your machine
- **No data transmission** - Your data never leaves your local environment
- **No telemetry or tracking** - We don't collect any usage data
- **No authentication required** - For local use only
- **Open source** - Full transparency, audit the code yourself

## Reporting a Vulnerability

If you discover a security vulnerability in Zapomni, please help us maintain the security of the project by following these guidelines:

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security issues via one of these methods:

1. **Preferred:** Create a private security advisory on GitHub:
   - Go to https://github.com/alienxs2/zapomni/security/advisories
   - Click "New draft security advisory"
   - Fill in the details

2. **Alternative:** Email the maintainer directly:
   - Email: Create an issue titled "Security: [Brief Description]" and we'll contact you privately

### What to Include

Please include as much of the following information as possible:

- **Type of vulnerability** (e.g., injection, authentication bypass, data exposure)
- **Full paths** of source file(s) related to the vulnerability
- **Location** of the affected source code (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact** of the vulnerability (what an attacker could do)
- **Suggested fix** (if you have one)

### Response Timeline

We are committed to responding to security reports promptly:

- **Initial response:** Within 72 hours
- **Status update:** Within 7 days with preliminary assessment
- **Resolution:** Depends on severity, but we aim for:
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next minor release

### Disclosure Policy

- We ask that you **do not publicly disclose** the vulnerability until we've had a chance to address it
- We will work with you to understand and resolve the issue
- Once fixed, we will:
  - Release a patched version
  - Publish a security advisory (with your permission)
  - Credit you for the discovery (if you wish)

## Security Best Practices for Users

### Safe Configuration

1. **Environment Variables**
   - Never commit `.env` files with real credentials
   - Use `.env.example` as a template only
   - Restrict `.env` file permissions: `chmod 600 .env`

2. **Docker Services**
   - Don't expose FalkorDB/Redis ports publicly
   - Use strong passwords for production
   - Keep Docker images updated

3. **Data Privacy**
   - Review data before storing in memory
   - Be cautious with sensitive information
   - Zapomni stores everything locally - secure your machine

### Secure Development

If you're contributing to Zapomni:

- Run `pre-commit install` to enable security checks
- Never commit secrets or credentials (even in tests)
- Use Pydantic `SecretStr` for sensitive configuration
- Follow the principle of least privilege
- Keep dependencies updated

### Dependencies

We actively monitor dependencies for known vulnerabilities:

- Automated dependabot alerts enabled
- Regular dependency updates
- Use `pip-audit` to check your installation:
  ```bash
  pip install pip-audit
  pip-audit
  ```

## Known Limitations

### Not Security Issues

The following are known limitations, not security vulnerabilities:

1. **Local machine security** - If your machine is compromised, Zapomni data is accessible
2. **File system permissions** - Data stored with standard user permissions
3. **No encryption at rest** - Data stored unencrypted in FalkorDB/Redis (local)
4. **No access control** - Single-user system, no multi-user authentication
5. **Ollama security** - We trust the local Ollama instance

These are intentional design decisions for a local-first tool. If you need enterprise security features (encryption, access control, audit logs), please open a feature request.

## Security Updates

Security updates will be released as:

- **Patch versions** (0.1.x) for fixes that don't break compatibility
- **Minor versions** (0.x.0) for fixes requiring minor changes
- **Security advisories** published on GitHub for critical issues

Subscribe to releases and security advisories:
- Watch the repository → Custom → Security alerts
- Star the repository to stay updated

## Bug Bounty

We currently do not offer a paid bug bounty program. However:

- We deeply appreciate security researchers' contributions
- Credit and recognition in release notes and CHANGELOG
- Acknowledgment in the security advisory
- Our gratitude and respect

## Security Hall of Fame

Contributors who responsibly disclose security issues:

- _No reports yet - you could be first!_

## Questions?

If you have questions about security that don't involve reporting a vulnerability:

- Open a discussion: https://github.com/alienxs2/zapomni/discussions
- Check existing issues: https://github.com/alienxs2/zapomni/issues

## Acknowledgments

This security policy is inspired by:
- [GitHub's Security Policy](https://github.com/github/docs/security/policy)
- [OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/)
- The broader security community

---

**Thank you for helping keep Zapomni and its users safe!**

Last updated: 2025-11-26
