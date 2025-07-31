#!/bin/bash

output_file="count.csv"
echo "dataset,method,base_model,level,num_jpg" > "$output_file"

# Traverse all datasets under eval_images/
for dataset in eval_images/*; do
    [ -d "$dataset" ] || continue
    dataset_name=$(basename "$dataset")

    for method in "$dataset"/*; do
        [ -d "$method" ] || continue
        method_name=$(basename "$method")

        for base_model in "$method"/*; do
            [ -d "$base_model" ] || continue
            base_model_name=$(basename "$base_model")

            has_level=0
            for level in "$base_model"/*; do
                if [ -d "$level" ]; then
                    has_level=1
                    level_name=$(basename "$level")
                    count=$(find "$level" -type f -iname "*.jpg" | wc -l)
                    echo "$dataset_name,$method_name,$base_model_name,$level_name,$count" >> "$output_file"
                fi
            done

            if [ "$has_level" -eq 0 ]; then
                count=$(find "$base_model" -type f -iname "*.jpg" | wc -l)
                echo "$dataset_name,$method_name,$base_model_name,none,$count" >> "$output_file"
            fi
        done
    done
done

echo "Statistics completed, results saved to $output_file"
