# Core literature library

The papers the Librarian searches first. `papers.csv` lists them; structured summaries (evidence-card inputs, no full texts) will be added per paper.

## Selection rule (ARCHITECTURE §5.6)

1. **Coverage.** The original paper for each factor students can use at the stock layer or are likely to ask about, plus the methods papers that Priors' own procedures rely on (factor models, multiple testing, replication, decay, costs, combination, sentiment).
2. **Ranking.** Citation count first, from OpenAlex, never from memory. Among papers of similar citation weight, newer papers come first.

Every row records the citation count, its source, and the retrieval date, so the ranking can be reproduced and refreshed.

## Verification

Each paper was looked up by DOI in OpenAlex, and its title, first author, and year were checked against the record. Three records needed manual confirmation because OpenAlex stores the online-first year or a variant title:
- Hou, Xue & Zhang (2015)
- Hou, Xue & Zhang (2020)
- DeMiguel, Garlappi & Uppal (2009)

Sloan (1996) has no DOI in the usual Crossref form; it was matched by title search.

## Columns

| Column | Meaning |
|---|---|
| `tier` | `methods` (procedures Priors uses) or `factor` (original evidence for a factor) |
| `role` | What the paper is used for in Priors |
| `factor_ids` | Matching ids in the factor library (Ken French, Hou-Xue-Zhang, or Chen-Zimmermann acronyms) |
| `cites`, `cites_source`, `cites_asof` | Citation count and where and when it was retrieved |
