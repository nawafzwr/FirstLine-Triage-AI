# FirstLine architecture

```mermaid
flowchart TD
    A[Enter patient details and complaint] --> B[Map text or select complaint categories manually]
    B --> C[Clinician confirms categories]
    C --> D[Enter vitals on the same page]
    D --> E[Request ESI suggestion]
    E --> F[Worker runs the exported ESI estimator]
    F --> G[Clinician accepts or changes the suggested ESI]
    G --> H[Browser queues the case by final ESI]
    H --> I[Call patient in]
```

The optional hosted mapper receives the complaint narrative. Scoring uses confirmed categories and structured intake fields. The model runs inside the Worker. The Python estimator is also included for reproduction. The queue lives in the browser. Review drafts are short-lived and session-bound.
