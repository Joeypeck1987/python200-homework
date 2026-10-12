# Week 2: Regression

Install the libraries from the repository root:

```text
python -m pip install -r assignments_02/requirements.txt
python assignments_02/warmup_02.py
python assignments_02/project_02.py
```

The project uses Charlotte, NC in 2023 and compares Phoenix, three years of
Charlotte weather, and two month encodings. Data comes from the
[Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api).
Downloaded CSVs are retained in `outputs/` for repeat runs and Week 3.
Delete the relevant cached CSV to download fresh data.

Predictions use measurements from the same day. This is a regression learning
exercise, not a future-weather forecast. Test sets contain randomly held-out
days; a future-year evaluation would require a chronological split.

AI assistance was used to prepare and verify these exercises. Review the code
and interpretations before submitting, especially the warmups intended for
independent practice.
