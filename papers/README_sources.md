# Paper sources under `papers/`

## arXiv LaTeX

Each arXiv entry is stored as:

- `arxiv_<id>_src/` — unpacked TeX tree (figures, styles, bibliography, and at least one `.tex` file).
- `arxiv_<id>_source.tar.gz` — optional tarball from arXiv for archival refresh (most ids have both; `2604.21809` currently has only `_src/`).

Identifiers present in this directory:

| arXiv id   | Unpacked tree              | Source tarball (if present)        |
|------------|----------------------------|------------------------------------|
| 1202.0622  | `arxiv_1202.0622_src/`     | `arxiv_1202.0622_source.tar.gz`    |
| 2011.13456 | `arxiv_2011.13456_src/`    | `arxiv_2011.13456_source.tar.gz`   |
| 2303.08797 | `arxiv_2303.08797_src/`    | `arxiv_2303.08797_source.tar.gz`   |
| 2408.03054 | `arxiv_2408.03054_src/`    | `arxiv_2408.03054_source.tar.gz`   |
| 2504.18506 | `arxiv_2504.18506_src/`    | `arxiv_2504.18506_source.tar.gz`   |
| 2506.13523 | `arxiv_2506.13523_src/`    | `arxiv_2506.13523_source.tar.gz`   |
| 2510.11923 | `arxiv_2510.11923_src/`    | `arxiv_2510.11923_source.tar.gz`   |
| 2602.06791 | `arxiv_2602.06791_src/`    | `arxiv_2602.06791_source.tar.gz`   |
| 2602.22122 | `arxiv_2602.22122_src/`    | `arxiv_2602.22122_source.tar.gz`   |
| 2603.18992 | `arxiv_2603.18992_src/`    | `arxiv_2603.18992_source.tar.gz`   |
| 2604.21809 | `arxiv_2604.21809_src/`    | *(none in repo)*                   |

**Policy:** arXiv-compiled PDFs are not kept under `papers/`; use the TeX tree (or re-fetch from arXiv) to regenerate a PDF locally if needed.

### Refresh or add an arXiv paper

From the repository root (replace `<id>` with the dotted id, e.g. `2510.11923`):

```bash
curl -L -o "papers/arxiv_<id>_source.tar.gz" "https://arxiv.org/e-print/<id>"
```

Inspect the archive type (`file papers/arxiv_<id>_source.tar.gz`) — it is usually gzip-compressed tar. Unpack into a clean directory:

```bash
rm -rf "papers/arxiv_<id>_src"
mkdir -p "papers/arxiv_<id>_src"
tar -xzf "papers/arxiv_<id>_source.tar.gz" -C "papers/arxiv_<id>_src"
```

If the archive unpacks with a single top-level folder, either move contents up one level or keep as-is; the requirement is that `find papers/arxiv_<id>_src -name '*.tex'` finds at least one file.

## PDF-only exceptions

These remain as PDFs because public author LaTeX is generally **not** distributed the same way as on arXiv:

- `biorxiv_*.pdf` — bioRxiv preprints.
- `Wang_et_al_2026_AnewSampling.pdf` — non-arXiv reference without a matching bundled TeX tree here.

## Other

- `PriceofFreedom/` — separate submodule / project material, not part of the arXiv paper bundle layout above.
