# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/vitoramaral10/whatsapp-mcp/security/advisories/new),
not in public issues. Expect an answer within a week. Only the latest version on `main` is
supported.

## Deployment model

This server holds a full WhatsApp session and message history, and neither the MCP endpoint nor
the REST API authenticates callers. It assumes it runs on a trusted network or behind an
authenticating proxy; see the Security section of the [README](README.md#security).
