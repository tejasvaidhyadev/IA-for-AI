python3 run_wav2vec_clf.py \
    --pooling_mode="mean" \
    --model_name_or_path="facebook/hubert-large-ll60k" \
    --model_mode="wav2vec" \
    --train_file="./../dataset/train_set.csv" \
    --validation_file="./../dataset/valid_set.csv" \
    --test_file="./../dataset/test_set.csv" \
    --per_device_train_batch_size=1 \
    --input_column=filename \
    --target_column=emotion \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --num_train_epochs=8.0 \
    --delimiter="comma" \
    --evaluation_strategy="steps" \
    --save_steps=500 \
    --eval_steps=500 \
    --logging_steps=100 \
    --save_total_limit=2 \
    --output_dir="output_dir" \
    --do_eval \
    --do_train \
    --do_predict \
    --fp16 \
    --freeze_feature_extractor \
    --text_path='text.json' \
    --bert_name='bert-base-uncased' \
    --overwrite_output_dir