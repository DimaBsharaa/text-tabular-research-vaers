# Tennis source clarification drafts — not sent

**Archived reference only; not an action plan. No outreach is part of the current workflow.**

Prepared October 15, 2026 for review. These are draft messages only; no email, form or other communication was sent. Public contact details are source-listed and have not been tested for delivery.

## Cornell corpus maintainers

Contact listed by the [ConvoKit tennis documentation](https://convokit.cornell.edu/documentation/tennis.html): Liye Fu, `lf383@cornell.edu`.

**Subject:** Tennis interviews corpus: original source IDs, availability and research reuse

Hello,

We are students preparing a noncommercial NLP project using the Tennis Interviews corpus. Our proposed task combines an interview from a previous event with historical player statistics to predict a strictly later match.

Could you clarify:

1. Were the original ASAP transcript URLs/IDs or collection timestamps retained, and can they be shared? We need evidence that the transcript version used was available before a later prediction cutoff.
2. Does any agreement cover academic model training, text embeddings and public redistribution of these transcripts? If so, could you point us to its terms and identify which rights it covers? We understand that permission for the toolkit or metadata may differ from rights to the underlying text.
3. Were corpus dates taken from the source transcript headings or from the matched Tennis-Data records? We found the documented possibility of automated linking errors, including a result-label conflict in interview 1499.
4. If public transcript redistribution is not permitted, is there an established way to release identifiers and a reconstruction script for academic replication?

We can provide a compact list of records if helpful. Thank you.

## ASAP Sports

Public [contact page](https://www.asapsports.com/contact.php): web contact form; `peter@asapsports.com` is also listed there. The appropriate recipient should be confirmed when sending.

**Subject:** Request for noncommercial academic use and publication-date clarification

Hello,

We are preparing a university NLP project using a bounded collection of historical tennis interview transcripts originally distributed with Cornell's Tennis Interviews corpus. The project deadline is October 26, 2026.

We would like to clarify separately whether you permit:

- Local noncommercial model training and evaluation on those transcripts, including pretrained text embeddings and limited target-aware fine-tuning.
- Public redistribution of selected transcript text as part of an attributed research CSV, and any applicable restrictions.
- If text redistribution is not permitted, release of source URLs/IDs, metadata and a reconstruction script for authorized users.

We also need to establish historical availability before later matches. Do the archive headings represent publication dates as well as interview dates? Is there a record of first public upload times or a documented historical publication policy applicable to 2007–2015?

We have not publicly released the interview text. Please let us know the permitted scope, attribution requirements and whether any other rights holder's authorization is required. Thank you.

## Tennis-Data

Use the contact route linked from the [source homepage](https://www.tennis-data.co.uk/); do not assume a contact address from an unrelated site.

**Subject:** Clarification of noncommercial academic ML use of historical tennis records

Hello,

We are preparing a university NLP project combining historical tennis statistics with prior interview text. Your current homepage restricts certain data-training uses, so we would like explicit clarification before proceeding.

Would you permit this noncommercial academic project to use a bounded set of historical 2007–2015 ATP/WTA records for model training and evaluation? Would permission also cover an attributed public CSV of selected/derived features, dates and outcomes, or would that require separate terms?

Please clarify the permitted scope and any attribution or redistribution conditions. We can keep source tables and interview transcripts separate in the release. Thank you.

## Project staff

**Subject:** Track 2 tennis candidate: evaluation protocol and dataset-release feasibility

We are considering a tennis task using earlier player interviews and historical statistics. We have not selected it or claimed curation success.

The official benchmark uses random holdouts. Our audit finds that later training examples can contain interviews about earlier held-out target matches, potentially exposing their outcomes. We plan to preserve the official code and report a separate chronological diagnostic; we are also assessing source/target-disjoint construction.

Would the project accept this combination of official curation results and a clearly labeled chronological evaluation? If transcript redistribution is not licensed, would a metadata/URL release with an authorized reconstruction workflow satisfy the dataset deliverable, or must we provide an openly redistributable text CSV?

We want to resolve these feasibility questions before spending GPU time or committing to the dataset.
