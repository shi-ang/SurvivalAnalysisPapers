# Repository Guidelines

## Project Structure & Module Organization

This repository is a curated survival-analysis reading list, not an executable software package. The main content lives in `README.md`, which contains the project description, topic badges, table of contents, and paper tables. There are currently no `src/`, `tests/`, build assets, or package configuration files.

Organize new entries under the topic that best matches the paper's primary contribution. If a paper spans multiple areas, choose the most representative section rather than duplicating it.

## Build, Test, and Development Commands

No build or runtime commands are required. Useful maintenance commands are:

```bash
git status --short
```

Shows local changes before committing.

```bash
rg "Paper Title" README.md
```

Checks whether a paper or resource is already listed.

```bash
git diff -- README.md AGENTS.md
```

Reviews documentation-only edits before opening a pull request.

## Markdown Style & Naming Conventions

Keep all repository documentation in Markdown. Use ATX headings (`## Section Name`) and maintain the existing table format in `README.md`.

For paper rows, follow the existing columns for that section, usually `Keyword`, `Title`, `Publisher`, `Date`, `Code`, and `Notes`. Use dates in `YYYY.MM` format when the month is known, or `YYYY` when only the year is available. Prefer canonical paper links such as publisher pages, OpenReview, arXiv, or proceedings pages. Add code links only when they point to an official or clearly associated implementation.

## Validation Guidelines

There is no automated test suite. Before submitting changes, preview the Markdown and verify that tables render correctly. Check that links are valid, topic anchors in the table of contents still match headings, and badge counts remain accurate if resources or topics are added.

## Commit & Pull Request Guidelines

Recent history uses short, direct commit messages such as `Update README.md`, `Add LLM curation note`, and date-based update messages. Keep commits focused on one logical documentation change.

Pull requests should describe what was added or changed, list any new topic sections, and mention whether badge counts were updated. For new papers, include enough citation metadata for review: title, venue or publisher, date, paper link, and optional code link.

## Agent-Specific Instructions

LLM-assisted contributions are welcome, but final curation should remain human-reviewed. Do not add papers solely from generated summaries; verify bibliographic details and links against primary sources before editing `README.md`.

### AI-Agent Research Discovery Workflow

Use AI agents as research assistants for discovery and triage, not as the final authority on inclusion. For "latest papers" tasks, agents should follow this pipeline before proposing any `README.md` edits:

1. **Search broadly but selectively.** Prioritize recent survival-analysis and time-to-event papers from machine learning venues and primary sources: arXiv, OpenReview, proceedings pages, publisher pages, official conference sites, and author project pages. Give highest priority to venues such as NeurIPS, ICML, ICLR, AISTATS, KDD, AAAI, UAI, CHIL, MLHC/ML4H, and relevant workshops.
2. **Include adjacent areas when valuable.** Consider statistics, biomedical informatics, clinical applications, and review papers when they make a clear survival-analysis contribution, introduce a useful method, evaluate survival models, or summarize an important subfield.
3. **Deduplicate before recommending.** Search `README.md` for exact and variant titles, acronyms, author/project names, and arXiv/OpenReview identifiers before proposing a candidate.
4. **Verify from primary sources.** Confirm title, venue or publisher, date, canonical paper link, and official code link if available. Do not rely only on search snippets, secondary summaries, or LLM-generated descriptions.
5. **Prepare a triage report first.** Before editing `README.md`, provide a candidate list with title, link, venue or publisher, date, paper type, likely README section, code link if official, and include/exclude rationale. Also include a short "not added" list for rejected, duplicate, or low-confidence papers.
6. **Edit only after curation.** Add papers only after the candidate list is reviewed or the user explicitly asks for README row additions. Place each accepted paper in the single section that best matches its main contribution.

When converting approved candidates into `README.md` rows, preserve the existing Markdown table format, use `YYYY.MM` dates when the month is known, prefer canonical paper links, and add code links only for official or clearly associated implementations. After edits, check that links are valid, table rows render correctly, topic anchors still match the table of contents, and topic/resource badge counts are updated if needed.

### AI-Agent README Audit Workflow

Use this workflow when agents are asked to examine or verify existing `README.md` entries. The goal is to find stale or incorrect metadata while keeping final curation human-reviewed.

1. **Audit existing rows systematically.** Check each selected entry for title, venue or publisher, date, paper link, code link, notes, section placement, and duplicate listings.
2. **Upgrade arXiv entries conservatively.** For rows linked to arXiv or marked `Arxiv`/`ArXiv`, search for a formally published version. If a verified proceedings, journal, OpenReview, publisher, DOI, or PMLR/ACM/IEEE/OUP/Nature/Springer/Taylor & Francis page exists, propose replacing the arXiv metadata with the published version.
3. **Prefer primary sources.** Verify corrections against primary sources rather than search snippets, citation aggregators, secondary summaries, or LLM-generated descriptions. Keep arXiv as the link when it remains the canonical source, when no published version can be verified, or when the arXiv page is intentionally more useful than an inaccessible publisher page.
4. **Check for metadata mismatches.** Flag wrong or stale venue names, publication dates, titles, broken or noncanonical links, typoed code labels, non-official code repositories, duplicate papers, and rows placed under a weak topic section.
5. **Prepare an audit report first.** Before editing `README.md`, report the current row, issue type, verified source, proposed correction, confidence level, and rationale. Group findings into `Update recommended`, `Needs human decision`, and `No change`.
6. **Edit only after approval.** Apply README changes only after the user approves the audit findings or explicitly asks for updates. For approved arXiv-to-published upgrades, update the title link, publisher, date, and official code link if available while preserving the existing table format.

After approved audit edits, search `README.md` by title, acronym, arXiv ID, DOI, and code repository name to catch duplicates. Recheck Markdown table rendering, link validity, topic anchors, date format, and topic/resource badge counts before submitting changes.
