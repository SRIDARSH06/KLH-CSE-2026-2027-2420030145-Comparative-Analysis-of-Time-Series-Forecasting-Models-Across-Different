---
name: Root preview routing
description: Preview routing behavior when a standalone Streamlit workflow shares an artifact workspace
---

The shared public root is controlled by the application router, not just by a healthy workflow port. A standalone Streamlit workflow needs an explicit root route and a bridge from the routed service to the Streamlit port.

**Why:** Without a root artifact route, the public root can resolve to the scaffold API and return 404 even when Streamlit is healthy on its local port.

**How to apply:** Preserve the validated root route and local Streamlit bridge when restarting or restructuring this project; keep `/api` and canvas paths available.