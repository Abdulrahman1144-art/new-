# Data dictionary

The workbook has 200,000 rows and 20 source columns. Names and definitions below
are inferred from the column headers and values; the provider has supplied no
authoritative dictionary.

| Field | Inferred meaning | Modeling use |
|---|---|---|
| Patient_ID | Record identifier | Excluded |
| Age | Age in years | Included |
| Gender | Recorded gender | Included |
| Nationality | Emirati / expatriate label | Included |
| Emirate | Emirate | Included |
| Diagnosis_Date | Diagnosis date | Excluded from baseline model |
| Cancer_Type | Cancer category | Included |
| Cancer_Stage | Recorded stage | Included |
| Treatment_Type | Treatment category | Excluded (post-diagnosis decision) |
| Treatment_Start_Date | Treatment start date | Excluded |
| Hospital | Hospital name | Excluded |
| Primary_Physician | Physician identifier/name | Excluded |
| Outcome | Recovered / Under Treatment / Deceased | Target (recorded label only) |
| Death_Date | Death date; missing for non-deceased records | Excluded |
| Cause_of_Death | Cause; missing for non-deceased records | Excluded |
| Smoking_Status | Smoking category | Included |
| Comorbidities | Recorded comorbidity; missing values retained as Unknown | Included |
| Ethnicity | Recorded ethnicity label | Included; assess fairness before any use |
| Weight | Weight (assumed kg; units unverified) | Used only to derive BMI, then excluded |
| Height | Height (assumed cm; units unverified) | Used only to derive BMI, then excluded |

## Quality observations

- 200,000 rows and 20 columns; zero exact duplicate rows and duplicate IDs.
- Missing cells: `Death_Date` 171,300; `Cause_of_Death` 171,300; `Comorbidities` 80,660.
- Death-field missingness is consistent with outcomes and is semantically meaningful; it is not imputed for modeling.
- Age ranges 18–92, weight 38–131, height 140–205 in this file; none breach configured broad plausibility bounds.
- Diagnosis and treatment dates parse successfully and treatment start is not before diagnosis for observed records.
- Category values are internally consistent in the observed file. There is no demonstrated inconsistent spelling or format variation to repair.
- Missingness and some outcome/stage associations do not establish real-world validity; the workbook source and generation method are unknown.
