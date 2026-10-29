---
name: Python environment
description: Replit Python runtime and package-installation constraint for this project
---

Use the full Python tools module with pip for this Streamlit project rather than the base Python module.

**Why:** The base Python module can expose an interpreter without pip and rejects package installation because the system environment is externally managed.

**How to apply:** If dependency installation reports missing pip or an externally managed environment, enable a full supported Python toolchain first, then install from requirements.txt.