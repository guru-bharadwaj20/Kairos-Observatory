# Governance and Independence Policy

KAIROS is a measuring instrument. An instrument is only useful if the people it
measures cannot adjust the reading. This document sets out who decides what, how
the project stays independent, and how its one philosophical parameter, the
Jonas loss ratio `λ`, is set.

The rules here follow the philosophy layer in
[KAIROS_Project_Idea.md](KAIROS_Project_Idea.md) (section 6) and the
non-negotiables in [CONTRIBUTING.md](CONTRIBUTING.md) (section 2.5).

---

## 1. Principles

| Principle | Source | Rule |
|---|---|---|
| No self-certification | Löb's theorem | No AI system and no lab may be the final judge of its own safety, or of KAIROS readings about it. |
| Published parameters | Jonas | `λ` and every tripwire threshold live in versioned YAML in this repository. |
| Falsifiability | Popper | Forecasts are pre-registered before outcomes are known. Ledger entries are never edited; corrections are appended. |
| Openness | Löb | Code, data, methods and forecasts are public. Anyone can rerun the pipeline and check a reading. |
| Recommend, not decide | Section 8 | KAIROS publishes tripwire levels and recommended responses. It never takes autonomous action. |

## 2. Roles

- **Maintainer** (currently [guru-bharadwaj20](https://github.com/guru-bharadwaj20)):
  merges changes, cuts releases, runs the weekly pipeline and enforces this policy.
- **Contributors:** anyone who opens an issue or pull request under the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- **Methods reviewers:** contributors with a track record in forecasting,
  statistics or AI evaluation who review changes to dial definitions,
  thresholds and validation. Any contributor may nominate one; the maintainer
  records appointments in `GOVERNANCE.md`.
- **Independent replicas:** third parties who run the pipeline on their own
  hardware and publish their readings. Divergence between replicas is reported
  on the dashboard.

As the project grows, the maintainer role passes to a steering group of at least
three people from different institutions, none of whom may be employed by a
measured lab (section 3).

## 3. Funding and independence

1. **No funding from measured labs.** KAIROS does not accept money, compute,
   credits, in-kind services or staff time from any organisation whose models
   are measured by the `C(t)` index (a "measured lab"), or from their
   controlling entities.
2. **No privileged data.** KAIROS does not use data that only a measured lab can
   verify. Every series must come from a public source or be reproducible by a
   third party.
3. **Disclosed funding.** All other funding (grants, donations above USD 1,000,
   sponsored compute) is listed publicly in `FUNDING.md` with the funder, amount
   and date, before the money is used.
4. **No editorial strings.** No funder may review, delay or veto a forecast,
   dial reading or publication.
5. **Walk-away rule.** If a funder becomes a measured lab, or is acquired by
   one, KAIROS stops accepting new money from it and discloses the change.

## 4. Conflicts of interest

Every maintainer, methods reviewer and contributor who changes a dial
definition, a threshold, `λ` or a validation result must disclose, in the pull
request:

- current or past (last 3 years) employment, consulting or board roles at a
  measured lab or AI policy organisation;
- equity or other financial interests above USD 10,000 in a measured lab;
- grants or funding that depend on KAIROS readings.

Disclosures are kept in `docs/conflicts.md`. A person with a conflict may
comment on a change but may not approve or merge it. Undisclosed conflicts are
grounds for removing review rights.

## 5. Setting `λ` through public deliberation

`λ` is the cost of a missed catastrophe relative to a false alarm. The alarm
fires when `P(danger) > 1/(1 + λ)`. It is a value judgement, not a fitted
parameter, so it is set in public:

1. **Proposal.** Anyone may open an issue labelled `lambda` proposing a value
   with a written rationale.
2. **Comment period.** The issue stays open for at least **30 days** and is
   announced on the dashboard and in the weekly release notes.
3. **Evidence.** The maintainer publishes how the tripwire history (backtests
   and live ledger) would have changed under the proposed value.
4. **Decision.** The maintainer, with the methods reviewers, records the
   decision and the reasons in `config/governor.yaml` and in a changelog entry.
   Dissenting views are summarised in the same entry.
5. **Prospective only.** A new `λ` applies from the next pre-registered
   forecast. Past ledger entries are never re-scored under a new `λ`, so
   nobody can tune the parameter to flatter the record.
6. **Review.** `λ` is reviewed at least once a year.

The same procedure applies to every tripwire threshold.

## 6. Changes to this policy

Changes to `GOVERNANCE.md` follow the `λ` procedure: a public issue, a 30-day
comment period and a recorded decision. Sections 3 and 5.5 cannot be weakened
while any forecast in the ledger is still unscored.

---

*KAIROS recommends; humans decide.*
