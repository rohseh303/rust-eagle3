# Publication copy versus execution record

The original execution record is preserved separately. This copy makes the following publication-only changes:

1. Replace device UUIDs with consistent H100-01 through H100-08 labels. Replace private mounted-volume paths with `/data`, local result-root prefixes with relative paths, and personal machine paths with a marker. Prompt strings, generated answers, timings, counters, selection data, and model bytes are unchanged.
2. Select completed result files and training audits. Exclude account billing detail, cloud-control-plane logs, credentials, failed/interrupted attempt logs, and large optimizer/feature files. Retain the completed pilot concurrency results, including output mismatches; do not erase negative findings.
3. Preserve historical protocol language and addenda as a dated record, including its original no-public-upload boundary. This prepared release is a later packaging step, not evidence that the original experiment was public.
4. Require an explicit cloud namespace in `modal_job.py` and cloud retrieval helpers. Require explicit initialization of a new budget ledger; do not inherit the experiment author's historical spending authorization. New experiments are created outside this evidence tree, without the previously passed replication gate.
5. Add offline verification, publication documentation, model cards, licensing notices, and CPU-only CI. These helpers were locally checked. No fresh GPU reproduction or clean image rebuild was performed for the release.

The numerical comparisons are reproduced from the distributed records by `scripts/verify_release.py`. `ARTIFACT_SHA256.json` covers data, results, audits, figures, source pins, model metadata, and the frozen protocols. Public model cards have their own checksums; original binary-weight checksums are preserved.
