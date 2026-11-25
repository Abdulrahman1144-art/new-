# Dataset selection and provenance log

## Supplied file

The project uses `uae_cancer_dataset_200k.xlsx`, supplied directly with the
project request. Inspection confirms 200,000 rows and 20 columns. The file has
a usable categorical label (`Outcome`) and UAE-oriented fields (`Emirate`,
`Nationality`). However, no publisher, URL, license, collection method, date
coverage statement, or evidence of real patient data was supplied.

## Search evidence

No external dataset search has been performed as part of this workspace task.
The supplied workbook was selected for technical exploration because it meets
the numeric size/structure requirement and has a target field. Its provenance
and license have not been verified. Before Milestone 2, the team should search
and record queries and decisions on Kaggle, UCI, Google Dataset Search, and UAE
government open-data portals; record the exact candidate URLs, licenses, and
reasons for acceptance/rejection. If this workbook is synthetic, obtain written
instructor approval before using it for the assessed project.

## Publication decision

Do not publish the workbook or claim it represents UAE cancer patients until
ownership, license, provenance, and privacy status are confirmed. The app and
README state this limitation. Public Streamlit deployment also requires a
publicly accessible, licensed dataset and committed model artifact or a safe
download workflow; neither is configured here.
