---
name: NN5 workbook variability
description: Practical compatibility rule for NN5 benchmark workbooks
---

Treat the configured NN5 worksheet name as a preferred location, not a guarantee; valid files may place the observation table on another worksheet and include metadata rows.

**Why:** NN5 files in circulation are not uniform, so strict worksheet-name assumptions make otherwise usable uploads fail.

**How to apply:** Keep worksheet fallback and metadata-aware date/series detection when changing the loader.