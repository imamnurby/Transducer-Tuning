```
python assert_generation.py \  
    --base_directory /path/to/base_directory \  # Specify the base directory where the input text files are located
    --splits valid test train \                 #  List the splits to process: 'valid', 'test', and 'train'
```

Make sure that you have downloaded the data from the [prior work](https://github.com/NougatCA/FineTuner). If you have not, then download the `train_assert.txt`, `train_methods.txt`, `valid_assert.txt`, `valid_methods.txt`, `test_assert.txt`,  and `test_methods.txt` and put those files in this directory. 