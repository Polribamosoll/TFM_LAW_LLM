
# Conditional generation
if extract_core_text:
    
    # Tokenizer
    core_tokenizer = transformers.AutoTokenizer.from_pretrained(
        core_model_id
    )
    
    # Set BNB configuration if quantization is enabled
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if use_quantization else None
    
    # Set model
    core_model = transformers.AutoModelForCausalLM.from_pretrained(
        core_model_id,
        trust_remote_code = True,
        quantization_config = bnb_config,
        device_map = "auto"
    )
    
    # Pipeline
    core_pipeline = pipeline(
        model = core_model,
        tokenizer = core_tokenizer,
        task = 'text-generation',
        model_kwargs = {"torch_dtype": torch.bfloat16},
        return_full_text = config["core_return_full_text"],
        max_new_tokens = config["core_max_new_tokens"],
        repetition_penalty = config["core_repetition_penalty"],
        temperature = config["core_temperature"],
        pad_token_id = core_tokenizer.eos_token_id,
        truncation = True,
        batch_size = 1
    )
    
    # Create object for loop
    splitted_df_v1_summary = splitted_df_v1.copy()

    # Initialize variables for token count and time measurement
    total_tokens_processed = 0
    start_time = time.time()

    # Iteration counter
    iteration_counter = 0

    # Loop by row to generate one summary per text
    for text_idx, core_context in tqdm(enumerate(splitted_df_v1_summary['text']), total=len(splitted_df_v1_summary)):
        
        # Generate summary
        summarized_text = generate_summary(core_pipeline, core_tokenizer, core_prompt, core_context)
        
        # Assign summary
        splitted_df_v1_summary.at[text_idx, 'core'] = summarized_text
        
        # Increment the iteration counter
        iteration_counter += 1
        
        # Check if it's time to perform garbage collection
        if iteration_counter % 1000 == 0:
            # Clean memory
            torch.cuda.empty_cache()
            gc.collect()

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Final format
    splitted_df_v1_summary = splitted_df_v1_summary.rename(
        columns={'text': 'original_text', 'core': 'text'}
    )

    # Save splitted & summarized data
    path = 'prepared_data/'
    csv_file_name_v1_summary = f'{path}splitted_input_core.csv'

    # Write the DataFrame to a CSV file
    splitted_df_v1_summary.to_csv(csv_file_name_v1_summary, index=False)

    # Show
    splitted_df_v1_summary.head()