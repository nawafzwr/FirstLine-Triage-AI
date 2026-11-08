# FirstLine data dictionary

These ranges are the demo input limits.

| Input | Accepted value | Model handling |
| --- | --- | --- |
| Age | 18–120 years, whole number | Numeric |
| Gender | Female or Male | Categorical |
| Arrival method | Ambulance or Walk-in | Ambulance indicator |
| Heart rate | 0–350 bpm | Numeric |
| Respiratory rate | 0–100 breaths/min | Numeric |
| Oxygen saturation | 0–100% | Numeric |
| Temperature | 20–50 °C | Numeric. Historical °F values were converted to °C. |
| Systolic blood pressure | 0–300 mmHg | Numeric |
| Diastolic blood pressure | 0–200 mmHg | Numeric |
| Chief complaint | English narrative, up to 500 characters | The model uses explicitly confirmed categories. Multiple labels are allowed. |
| Pain score | Optional, 0–10 | UI field only |

At least five of the six vital measurements must be entered. Missing values use training-fitted imputation and missingness indicators. The full complaint vocabulary is in the [model manifest](../model/manifest.json). The target is the historical recorded ESI level from 1 to 5.
