# Low-Resources-Machine-Translation
Implementing a Machine translation system from English to French for the course IFT-6759 


1. To train a en-fr model.
```
python train.py --epoch=10 --seed=1234 
```

2. To train a reverse(fr-en) model and generate files for back-translation
```
python train.py --epoch=10 reverse_translate=True generate_samples=True --seed=1234  
```

3. To use these additional files created above.
```
python train.py --epoch=10 add_synthetic_data=True --seed=1234  
```

__Configure config.py for more control over all parameters and paths.__ 


**Steps for the evaluation:**
```
python evaluator.py --input-file-path="path_to_en_file.txt" --target-file-path="path_to_org_fr_file.txt"

```
