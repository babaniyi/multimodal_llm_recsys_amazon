# Config Target Size Fix

## Problem Identified

The `balance_categories` method had a hardcoded default parameter of `400000` instead of reading from the train_config file.

## Issue

### Before Fix
```python
def balance_categories(self, all_data: Dict[str, List[Dict]], 
                      target_size: int = 400000) -> Dict[str, List[Dict]]:
```

- ❌ **Hardcoded default**: `target_size: int = 400000`
- ❌ **Not configurable**: Couldn't change dataset size without code modification
- ❌ **Inconsistent**: Default didn't match actual config usage

### After Fix
```python
def balance_categories(self, all_data: Dict[str, List[Dict]], 
                      target_size: int) -> Dict[str, List[Dict]]:
```

- ✅ **No hardcoded default**: `target_size: int` (required parameter)
- ✅ **Config-driven**: Always reads from `train_config.json`
- ✅ **Consistent**: Uses same config value throughout

## How It Works

### Config Reading
```python
# In run_complete_processing() method
if self.category_balance:
    target_size = self.config['model_config'].get('test_size', 100000)
    all_data = self.balance_categories(all_data, target_size)
```

### Config Value
```json
{
  "model_config": {
    "test_size": 400000,        // Total dataset size
    "min_dataset_size": 400000, // Minimum required size
    "max_dataset_size": 500000  // Maximum allowed size
  }
}
```

### Per-Category Calculation
```python
# In balance_categories() method
num_categories = len(all_data)  # 4 categories
target_per_category = target_size // num_categories  # 400000 ÷ 4 = 100000
```

## Benefits

### ✅ Configurable Dataset Size
- **Easy to change**: Modify `test_size` in config file
- **No code changes**: Update config without touching code
- **Flexible**: Can set any size within min/max limits

### ✅ Consistent Behavior
- **Single source of truth**: Config file controls dataset size
- **Predictable**: Same config value used everywhere
- **Maintainable**: Clear configuration management

### ✅ Proper Documentation
- **Updated docstring**: Reflects that target_size is required
- **Clear comments**: Shows config is the source of truth
- **Better understanding**: Makes the flow clearer

## Current Configuration

With `test_size: 400000`:
- **Total interactions**: 400,000
- **Per category**: 100,000 interactions each
- **Categories**: Appliances, Digital_Music, Gift_Cards, Health_and_Personal_Care

## Usage Examples

### Change Dataset Size
```json
{
  "model_config": {
    "test_size": 200000,  // 50k per category
    "min_dataset_size": 200000,
    "max_dataset_size": 500000
  }
}
```

### Increase Dataset Size
```json
{
  "model_config": {
    "test_size": 500000,  // 125k per category
    "min_dataset_size": 400000,
    "max_dataset_size": 500000
  }
}
```

## Result

The dataset size is now fully configurable through the `train_config.json` file, making it easy to:
- ✅ **Adjust dataset size** for different experiments
- ✅ **Scale up/down** based on available resources
- ✅ **Maintain consistency** across different runs
- ✅ **Document configuration** clearly

The fix ensures that the multi-category loader always respects the configured dataset size, making the system more flexible and maintainable.

