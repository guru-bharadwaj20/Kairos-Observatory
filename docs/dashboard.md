# Public dashboard (Phase 11)

The dashboard renders one **published snapshot** and nothing else. It never fits
models or reads raw data, so the public page is reproducible from a single file.

## Running it

```bash
uv sync --extra dashboard
uv run kairos dashboard serve --snapshot path/to/snapshot.json   # Streamlit app
uv run kairos dashboard export --snapshot path/to/snapshot.json --out site/
```

Until the `kairos` CLI mounts the command (`kairos.dashboard.cli.register`),
`python -m kairos.dashboard serve|export` does the same. Without `--snapshot`
the app reads `$KAIROS_SNAPSHOT`, and failing that the bundled demo fixture,
which is labelled **DEMO DATA** on every page.

## Snapshot schema (version 1)

The full field list lives in the docstring of `src/kairos/dashboard/snapshot.py`,
which is the source of truth. In short:

| Field | Producer | Content |
|---|---|---|
| `schema_version`, `generated_at`, `data_vintage`, `is_demo` | K-12.01 | Provenance |
| `parameters` | K-8.02, K-8.08 | `lambda`, `tripwires_version`, `vinge_tolerance_factor` |
| `alert` | K-8.05, K-8.08 | `level` 0-4, optional `name`/`response`, `lead_time` (median, lower, upper, interval, unit) or null |
| `dials` | K-8.01 | Exactly four `DialReading`s: `dial`, `symbol`, `value`, `unit`, `state` (`calm`/`danger`), `p_danger`, `history` |
| `capability`, `human_capacity` | K-7.01 | `SeriesFrame`s `{name, unit, t, value}`, base-year normalised (D8) |
| `acceleration` | K-3.04, K-5.04 | Per series: `t`, `observed`, `fit`, `lower`, `upper`, `changepoints`, `delta_samples` |
| `vinge_horizon` | K-6.04 | `t`, `h_star` at the published tolerance, optional `by_tolerance` |
| `ledger` | K-10.06 | Entries: `id`, `registered_at`, `statement`, `hash`, `prev_hash`, `proof` (`verified`/`pending`/`failed`), `outcome_date`, `score` |

Unknown extra keys are ignored, so producers can add fields without breaking the
dashboard. Anything that changes the meaning of an existing field needs a new
`schema_version`.

## Static export and GitHub Pages

`kairos dashboard export` writes `index.html` (one self-contained file with
plotly.js inlined once and no third-party requests), a copy of the snapshot as
`snapshot.json`, and `.nojekyll`. The weekly pipeline (K-12.01) publishes that
directory to GitHub Pages.

## Accessibility

- Okabe-Ito colours, which stay distinct under common colour-vision deficiencies.
- Colour is never the only cue: levels and dial states also carry words and symbols.
- `C(t)` and `H(t)` keep fixed colours and line styles (solid vs dashed) everywhere,
  and are only ever drawn together (§13).
- The banner is an ARIA live region; the static page has a skip link, landmarks and
  table headers.
