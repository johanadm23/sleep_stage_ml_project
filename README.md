# sleep_stage_ml_project
Predicting Sleep Stages from EEG Brainwave Features
## Sources & Acknowledgements

This project uses the Sleep-EDF dataset from PhysioNet:
- Goldberger AL, et al., "Sleep-EDF Database Expanded", PhysioNet. ([link](https://physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip))

Inspiration & helper resources:
- predict-idlab/sleep-linear — used as reference for dataset subset selection and recommended experimental splits. I implemented my own data loader and feature extractor; functions are original and credit is noted in the code where applicable.
- MNE-Python tutorials — used as reference for EDF reading and epoching; all usage is reimplemented and adapted.

If portions of code were adapted, relevant notes are included at the top of the file where the adaptation occurs.
