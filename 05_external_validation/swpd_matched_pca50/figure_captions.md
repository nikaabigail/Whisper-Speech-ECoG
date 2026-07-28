# Figure captions

**Figure 1. Matched acoustic-representation decoding.** Each open circle is one confirmatory patient (sub-02 through sub-09). Diamonds show the patient mean and error bars show two-sided 95% t confidence intervals. Neural inputs, temporal splits, train-only PCA50 transforms, OLS decoder, and metric were identical across MEL80 and Whisper layers.

**Figure 2. Patient-level paired comparison.** Lines connect the four target representations within each patient. The black line and diamonds show patient means. The within-patient design isolates the target representation while holding the ECoG data and decoder protocol fixed.

**Figure 3. Whisper-minus-MEL80 paired effects.** Points are patient-level differences. Diamonds and bars show mean differences and 95% t confidence intervals. Reported p-values are two-sided paired t-tests with Holm correction across the three predeclared layer-versus-MEL contrasts.

**Figure 4. Matched SWPD architecture.** ECoG high-gamma features and each acoustic target are reduced using fold-train-only standardized PCA50 transforms. A shared neural reducer and identical OLS decoder are used for all targets. Layer-specific PCA coordinates are not directly averaged, so this experiment does not define an L3+L4+L5 ensemble.

**Figure 5. Cohort accounting.** Ten participants were planned. sub-01 was used only for development. sub-10 was excluded because its official recording ends after 95 valid word trials and the final five event rows have zero duration at the final recorded sample. No imputation or participant-specific split was used. Primary inference therefore includes eight confirmatory patients.
