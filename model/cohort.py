"""Fixed input-availability rule for the next ESI research training cohort."""
VITAL_COLUMNS = [
    'triage_vital_hr', 'triage_vital_sbp', 'triage_vital_dbp',
    'triage_vital_rr', 'triage_vital_o2', 'triage_vital_temp',
]


def cohort_mask(raw):
    """Select recorded ESI 1–5 and at least five of six exported vitals.

    Apply before imputation. This is a research cohort definition, not a
    clinical determination of stability, bypass status or sufficient inputs.
    It does not modify rows or remove any particular recorded ESI level.
    """
    required = ['esi'] + VITAL_COLUMNS
    absent = set(required) - set(raw.columns)
    if absent:
        raise ValueError(f'Missing cohort-definition columns: {sorted(absent)}')
    valid_label = raw['esi'].isin([1, 2, 3, 4, 5])
    invalid = raw['esi'].notna() & ~valid_label
    if invalid.any():
        raise ValueError('Unexpected nonmissing ESI values require investigation.')
    return valid_label & raw[VITAL_COLUMNS].isna().sum(axis=1).le(1)
