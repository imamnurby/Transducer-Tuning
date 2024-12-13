# Instruction:
1. Prepare the raw dataset. 
    - Go to `prepare_new_data`
    - There are 4 directories: `code_summarization`, `assert_generation`, `code_translation`, `code_repair`. Each corresponds to the target task. Select the relevant directory. 
    - There is a README.md inside each directory. Execute the command attached in the `README.md.`

2. Execute `generate_dot.py`
```
python process_csv_script.py path/to/your/data.csv \  # Path to the CSV file to be processed
    --start 0 \                                       # Start processing from the first row (index 0)
    --end 100 \                                       # Stop processing at row 100 (end index is exclusive)
    --task_name summarization                         # Name of the target task
    --save_java \                                     # Flag to save the temporary Java files generated during processing
    --multiprocessing \                               # Flag to enable multiprocessing for generating dot files
    --num_workers 10 \                                # Number of worker threads to use for multiprocessing (set to 10)
    --output_filename my_output                       # Prefix for the output filenames (e.g., my_output.csv, my_output_preview.csv)
```

3. Execute `generate_node_vectors.py`
```
python generate_node_vectors.py \  
    --csv_path /data/spt/datasets/bugfix/50/processed_0_END.csv \               # Path to the input CSV file to be processed
    --base_path_dot_files /data/spt/datasets/bugfix/50 \                        # Base directory where the .dot files are located
    --output_path /data/spt/datasets/bugfix/50/processed_codet5p \              # Directory where the processed data will be saved
    --node_feature_path /data/spt/datasets/bugfix/50/node_features \            # Directory to save the extracted node features
    --edge_indice_path /data/spt/datasets/bugfix/50/edge_indices \              # Directory to save the extracted edge indices
    --model_path_for_node_feature mixedbread-ai/mxbai-embed-large-v1 \          # Path to the model used for extracting node features
    --model_path_for_backbone Salesforce/codet5p-220m \                         # Path to the backbone model used for source code tokenization
    --max_length 400 \                                                          # Maximum token length for the tokenizer (longer sequences will be truncated if needed)
    --truncation True \                                                         # Enable truncation of sequences that exceed the maximum length
    --start 0 \                                                                 # Start processing the DataFrame from row index 0
    --end 1000 \                                                                # Stop processing the DataFrame at row index 1000 (exclusive)
    --extract_node_features True \                                              # Flag to enable extraction of node features
    --num_workers 4 \                                                           # Use 4 worker threads for data loading and processing
    --batch_size 512 \                                                          # Process the data in batches of 512 samples at a time
    --node_embedding_size 1024                                                  # Size of the node embeddings (dimensionality of the feature vectors)

```

4. Execute `generate_dataset_dict.py`
```
python generate_dataset_dict.py \ 
    --input_dir processed_codet5p/ \                                    # Specify the directory where the dataset is currently stored
    --output_dir final_codet5p/ \                                       # Specify the directory where the final processed dataset will be saved
    --csv_path /data/spt/datasets/bugfix/50/processed_0_END.csv \       # Path to the CSV file containing the split indices for train, eval, and test sets
    --num_proc 10 \                                                     # Use 10 processes in parallel to speed up mapping, filtering, and tensor loading operations
    --output_filename final_codet5p                                     # Specify the name for the final output dataset directory

```

5. Execute `filter_dataset.py`
```
python filter_dataset.py \  
    --input_dir final_codet5p/ \            # Specify the directory where the dataset is currently stored
    --output_dir codet5_dataset_ready/ \    # Specify the directory where the filtered dataset will be saved
    --max_input_length 400 \                # Set the maximum length of 'input_ids' to 400 for filtering
    --max_node_features 50 \                # Set the maximum number of nodes to 50 for filtering
    --num_proc 10                           # Use 10 processes in parallel to speed up the filtering process

```

6. Execute `decontaminate.py`
```
python decontaminate.py \                       
    --dataset_path codet5_dataset_ready_v5/ \   # Specify the path to the dataset directory
    --output_csv included_ids.csv \             # Specify the output path for the cleaned test indices
    --tokenizer_name Salesforce/codet5p-220m \  # Use the 'Salesforce/codet5p-220m' tokenizer to decode 'input_ids'
    --num_perm 128 \                            # Set the number of permutations for the MinHash algorithm to 128
    --threshold 0.8                             # Set the LSH threshold to 0.8, which determines how similar two items must be to be considered duplicates

```