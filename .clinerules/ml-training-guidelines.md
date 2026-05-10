## Brief overview
  Project-specific guidelines for developing deep learning training pipelines in the RehabAI project. Covers LSTM model training for clinical score prediction on KiMoRe dataset, cross-validation methodology, and data integrity practices. Written for a backend engineer specializing in rehabilitation systems and deep learning.

## Language and communication
  - Respond in Vietnamese when the user writes in Vietnamese
  - Provide comprehensive analysis before proposing changes — cite sources (papers, docs, sklearn/Keras docs)
  - Present tradeoffs explicitly (e.g., training time vs. metric stability)
  - Use structured docs with markdown tables for change summaries
  - Always create a fix/plan document (e.g., `docs/fix_*.md`) before implementing changes to training pipelines

## Cross-validation and data integrity
  - Always use subject-level splitting (GroupKFold / StratifiedGroupKFold) — never leak subjects across train/val
  - Augmented data must only appear in train folds, never in validation
  - Fit StandardScaler on train fold only, then apply to val fold — never fit on full dataset before splitting
  - When fold score distributions are imbalanced (low val std), log warnings and consider excluding from OOF aggregation
  - Use quantile-based binning (≥5 bins) for stratification on skewed clinical score distributions
  - Repeated CV across multiple seeds is preferred when dataset is small (N<100) to reduce metric variance
  - Report per-fold diagnostics: y_val mean/std, score bin distribution, Wasserstein distance between train/val

## Model architecture and training
  - Use bounded output (sigmoid) when target is normalized to [0,1] range
  - CCC loss (1 - CCC) is preferred for clinical agreement tasks over MSE/Huber
  - Implement LR warmup callback to prevent early collapse in small-dataset training
  - Monitor prediction std per fold — flag collapse if pred_std < 0.5
  - Clip predictions to valid range after denormalization
  - Normalize y labels to [0,1] before training (e.g., y/50 for 0-50 score range)

## Metrics and evaluation
  - Primary metrics: Spearman ρ (rank correlation), CCC (concordance correlation coefficient)
  - Secondary: MAE, RMSE, Pearson r
  - Compare against paper baselines when available (e.g., KiMoRe baseline ρ values)
  - OOF predictions are the primary evaluation — final model trained on all data is for deployment only
  - Flag folds where y_val.std() < 2.0 as potentially unreliable for Spearman/CCC

## Code conventions
  - Python with type hints where practical
  - Use argparse for all CLI parameters with sensible defaults
  - TensorFlow/Keras for model building — prefer Functional API (Model class)
  - Logging: print fold-level diagnostics during training, save JSON results to plots directory
  - Save model config as JSON alongside .keras model and .joblib scaler
  - Use `MASK_VALUE = -999.0` for padded sequences with Masking layer

## Documentation standards
  - Include Vietnamese comments explaining non-obvious design decisions
  - Document temporal downsampling rationale (fps reduction, sequence length constraints)
  - Reference academic sources (arXiv, sklearn docs) in code comments and plan documents
  - Maintain architecture evolution docs (e.g., LSTM_ARCHITECTURE.md, LSTM_EVOLUTION.md)

## Other guidelines
  - Follow CLAUDE.md behavioral guidelines (surgical changes, simplicity first, goal-driven execution)
  - Before modifying training pipeline, create a plan document in `docs/` with proposed changes, rationale, and verification plan
  - Always ask for user review before implementing changes that affect training time (e.g., repeated CV, additional diagnostics)
  - Small dataset warning: with ~67 subjects, statistical reliability is limited — always report confidence/reliability caveats