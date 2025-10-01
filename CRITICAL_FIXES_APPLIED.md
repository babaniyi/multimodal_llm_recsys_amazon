# Critical Fixes Applied to Multimodal LLM RecSys

## Date: October 1, 2025

## Problems Identified and Fixed

### 1. **Input Format Missing Item Context**

**Problem**: The model was trained with only `<|user_ID|>` as input, providing no context about what the user purchased.

**Fix**: Updated `src/data/prepare_data.py` to include item details in the input:
```python
Input format: <|user_ID|> Item id is <|ASIN_XXX|>. Rating of the item by user from 1 to 5 is X.X. Review title: XXX. Item name is XXX. Main category of the item is XXX.
```

**Inspiration**: Matched the approach from the successful [LLMs-for-RecSys project](https://github.com/babaniyi/LLMs-for-RecSys/tree/main).

---

### 2. **Poor Data Quality - Too Many EOS-Only Samples**

**Problem**: 
- 23.4% of training samples had ONLY `<|endoftext|>` tokens (no actual next items)
- Average items per sample: only 2.68 (when asking for 10)
- This caused the model to learn that "most of the time, just predict EOS"

**Fix**: 
1. Increased `min_interactions` from 8 to 10
2. Modified `get_next_items()` function in `src/data/processor.py` to only process items where we have enough future interactions:
   ```python
   for i in range(num_interactions - number_of_items_to_predict):
   ```

**Result**: 
- **100% of samples now have actual items** (vs. 23.4% EOS-only before)
- Every sample has exactly 10 items

---

### 3. **Train/Test Split Causing Cold-Start Problem**

**Problem**: 
- Global temporal split was used (all old data in train, all new data in test)
- This caused **78.1% of test items to NEVER appear in training**
- The model had zero chance of predicting items it had never seen

**Fix**: Changed from global temporal split to **per-user temporal split** in `src/data/processor.py`:
```python
# Split PER USER to ensure item overlap
for user_id, user_data in df.groupby('user_id'):
    user_sorted = user_data.sort_values('date').reset_index(drop=True)
    n = len(user_sorted)
    
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))
    
    train_list.append(user_sorted.iloc[:train_end_idx])
    val_list.append(user_sorted.iloc[train_end_idx:val_end_idx])
    test_list.append(user_sorted.iloc[val_end_idx:])
```

**Result**: 
- Item overlap improved from **21.9% to 70.5%**
- Model can now predict items it has seen during training

---

### 4. **Missing model_response_items Field in Evaluation**

**Problem**: The `process_outputs.py` script wasn't creating the `model_response_items` field that the evaluation expects, causing all metrics to return 0.

**Fix**: Updated `src/evaluation/process_model_outputs.py` to store the items list:
```python
entry['model_response_items'] = model_response_items
```

---

## Summary of Improvements

| Metric | Before | After |
|--------|--------|-------|
| **Training Samples** | 2,113 | 2,488 |
| **EOS-only samples** | 23.4% | 0.0% |
| **Average items per sample** | 2.68 | 10.00 |
| **Input format** | User ID only | User ID + Item details |
| **Item overlap (train/test)** | 21.9% | 70.5% |
| **Repetitive predictions** | 53.7% | TBD (retraining) |

---

## Next Steps

1. ✅ Complete current training run with fixed data
2. ⏳ Run evaluation pipeline
3. ⏳ Compare with baselines
4. ⏳ Analyze if performance improved

---

## Key Learnings

1. **Input context is critical**: Models need rich item information to learn patterns
2. **Data quality matters**: Removing EOS-only samples improves training
3. **Split strategy is crucial**: Per-user splits prevent cold-start problems
4. **Evaluation pipeline must match data format**: Missing fields cause silent failures

---

## References

- Original LLMs-for-RecSys project: https://github.com/babaniyi/LLMs-for-RecSys/tree/main
- This project builds on that foundation by adding multimodal (image) features

