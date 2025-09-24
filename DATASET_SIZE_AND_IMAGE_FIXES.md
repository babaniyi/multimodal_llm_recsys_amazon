# Dataset Size and Image Processing Fixes

## Changes Made

### 1. Increased Dataset Size to 400k (100k per category)

#### Configuration Updates (`configs/train_config.json`)
```json
{
  "model_config": {
    "test_size": 400000,        // Increased from 100000
    "min_dataset_size": 400000, // Increased from 100000
    "max_dataset_size": 500000  // Kept same
  }
}
```

#### Multi-Category Loader Updates (`src/data/multi_category_loader.py`)
- ✅ **Target size**: Updated `balance_categories` default to 400000
- ✅ **Per-category**: Automatically targets 100k per category (400k ÷ 4 categories)
- ✅ **Balancing**: Maintains equal representation across categories

### 2. Fixed Image Processing Issues

#### Problem Identified
- ❌ **URL vs File Path**: Dataset was trying to use URLs as local file paths
- ❌ **High Resolution**: Using `hi_res` images which are larger and slower to download
- ❌ **No Validation**: Including entries without downloadable images

#### Solutions Implemented

##### A. Dataset Image Handling (`src/data/dataset.py`)
```python
# Handle URL images by downloading them first
if image_path.startswith('http'):
    # Download image to cache
    from .processor import download_and_cache_image
    cache_dir = Path("data/processed/image_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = download_and_cache_image(image_path, cache_dir)
    if local_path:
        image_tensor = self.multimodal_encoder.process_image(local_path)
    else:
        image_tensor = None
else:
    # Local file path
    image_tensor = self.multimodal_encoder.process_image(image_path)
```

##### B. Medium Resolution Images (`src/data/multi_category_loader.py`)
```python
def _get_downloadable_images(self, images: List[Dict]) -> List[str]:
    downloadable_images = []
    
    for img in images:
        # Try medium resolution first, then fall back to high resolution
        image_url = img.get('med_res') or img.get('hi_res') or img.get('low_res')
        
        if image_url and self._is_downloadable_url(image_url):
            downloadable_images.append(image_url)
            break  # Only take the first valid image
    
    return downloadable_images
```

##### C. Image URL Validation (`src/data/multi_category_loader.py`)
```python
def _is_downloadable_url(self, url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    
    # Check for common image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    url_lower = url.lower()
    
    # Must have a valid extension
    if not any(ext in url_lower for ext in valid_extensions):
        return False
    
    # Must be from a trusted domain
    trusted_domains = ['amazon.com', 'media-amazon.com', 'images-amazon.com']
    if not any(domain in url_lower for domain in trusted_domains):
        return False
    
    return True
```

##### D. Filter Entries with Valid Images (`src/data/multi_category_loader.py`)
```python
# Get downloadable images
image_urls = self._get_downloadable_images(meta.get('images', []))

# Only include entries with downloadable images
if not image_urls:
    continue
```

##### E. Processor Medium Resolution (`src/data/processor.py`)
```python
for img in images:
    if img.get('variant') == 'MAIN':
        # Prefer medium resolution, fall back to high resolution
        main_image_url = img.get('med_res') or img.get('hi_res')
        break
```

## Benefits

### ✅ Dataset Size
- **400k total interactions** (100k per category)
- **Balanced representation** across all 4 categories
- **Sufficient data** for robust training and evaluation

### ✅ Image Processing
- **Medium resolution images** for faster download and processing
- **Automatic URL downloading** with caching
- **Image validation** to ensure only downloadable images are included
- **Error handling** for failed image downloads
- **Trusted domain filtering** for security

### ✅ Performance Improvements
- **Faster image processing** with medium resolution
- **Reduced memory usage** with smaller images
- **Better error handling** prevents crashes
- **Caching system** avoids re-downloading images

## Expected Results

### Dataset Statistics
- **Total size**: 400,000 interactions
- **Per category**: ~100,000 interactions each
- **Categories**: Appliances, Digital_Music, Gift_Cards, Health_and_Personal_Care
- **Image coverage**: Only entries with downloadable images

### Image Processing
- **Resolution**: Medium resolution (med_res) preferred
- **Fallback**: High resolution (hi_res) if medium not available
- **Validation**: Only Amazon domain images included
- **Caching**: Images downloaded once and cached locally

## Testing

The fixes address the original errors:
- ✅ `KeyError: 'instruction'` - Fixed with data format compatibility
- ✅ `No such file or directory: 'https://...'` - Fixed with URL downloading
- ✅ Dataset size - Increased to 400k (100k per category)
- ✅ Image resolution - Using medium resolution for better performance

## Usage

The workflow now runs with:
```bash
python run_multimodal_workflow.py
```

This will:
1. Download 400k interactions (100k per category)
2. Process only entries with downloadable images
3. Use medium resolution images for faster processing
4. Cache images locally to avoid re-downloading
5. Run complete cross-category analysis

The training pipeline should now work without image processing errors and with the desired dataset size.


