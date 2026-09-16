# Evaluation datasets

`monetary_policy.jsonl` contains 30 benchmark cases for the public monetary-policy
corpus used during development. It stores questions, short reference answers, and
source/page identifiers rather than copied document passages.

Each downloader should adapt `template.jsonl` to the files they place in
`data/raw`. An answerable case needs at least one `gold_targets` entry. Pages and
chunk IDs are optional refinements; a target with only a filename evaluates at
source level. Unanswerable cases must use an empty target list.

Do not commit evaluation outputs that contain private answers or corpus details.
The generated `reports/` directory is ignored by Git.
