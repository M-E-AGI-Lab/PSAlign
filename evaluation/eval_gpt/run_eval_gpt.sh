#!/bin/bash

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"  # Change to the directory where the script is located

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Display help information
show_help() {
    echo -e "${BLUE}GPT Image Evaluation Tool Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode         Evaluation mode: evaluate (single evaluation) or compare (comparison evaluation)"
    echo "  -d, --dataset      Dataset to evaluate: sage_seen, sage_unseen, or all"
    echo "  -t, --test         Enable test mode, process only a small number of samples"
    echo "  -s, --sample-size  Number of samples per model in test mode (default: 10)"
    echo "  -r, --max-retries  Maximum retry count for API request failures (default: 3)"
    echo "  --timeout          Maximum timeout for API requests (seconds) (default: 120)"
    echo "  --debug            Enable debug mode"
    echo "  --models           List of models to evaluate, separated by spaces (default: base psa)"
    echo "  --model-a          First model name for comparison (default: base)"
    echo "  --model-b          Second model name for comparison (default: psa)"
    echo "  --model-a-name     Display name for model A (default is the same as --model-a)"
    echo "  --model-b-name     Display name for model B (default is the same as --model-b)"
    echo "  -h, --help         Display this help information"
    echo ""
    echo "Examples:"
    echo "  $0 --mode evaluate --dataset sage_seen --test"
    echo "  $0 --mode evaluate --dataset sage_seen --models base safetydpo psa"
    echo "  $0 --mode compare --dataset all --timeout 180"
    echo "  $0 --mode compare --dataset sage_seen --model-a esdu --model-b psa --model-a-name ESDU"
    echo ""
}

# Default parameters
MODE="compare"
DATASET="all"
TEST_MODE=""
SAMPLE_SIZE="10"
MAX_RETRIES="3"
DEBUG_MODE=""
TIMEOUT="120"
MODEL_A="base"
MODEL_B="psa"
MODEL_A_NAME=""
MODEL_B_NAME=""
MODELS_TO_EVALUATE="base psa"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -t|--test)
            TEST_MODE="--test"
            shift
            ;;
        -s|--sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -r|--max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="--debug"
            shift
            ;;
        --models)
            shift
            MODELS_TO_EVALUATE=""
            # Collect model names until another argument starting with '--' is encountered
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS_TO_EVALUATE="$MODELS_TO_EVALUATE $1"
                shift
            done
            MODELS_TO_EVALUATE="${MODELS_TO_EVALUATE## }" # Remove leading space
            ;;
        --model-a)
            MODEL_A="$2"
            shift 2
            ;;
        --model-b)
            MODEL_B="$2"
            shift 2
            ;;
        --model-a-name)
            MODEL_A_NAME="$2"
            shift 2
            ;;
        --model-b-name)
            MODEL_B_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate mode argument
if [[ "$MODE" != "evaluate" && "$MODE" != "compare" ]]; then
    echo -e "${RED}Error: Invalid evaluation mode '$MODE'${NC}"
    echo "Available modes: evaluate, compare"
    exit 1
fi

# Validate dataset argument
if [[ "$DATASET" != "sage_seen" && "$DATASET" != "sage_unseen" && "$DATASET" != "all" ]]; then
    echo -e "${RED}Error: Invalid dataset '$DATASET'${NC}"
    echo "Available datasets: sage_seen, sage_unseen, all"
    exit 1
fi

# Build the command
if [[ "$MODE" == "evaluate" ]]; then
    CMD="python evaluate_images.py"
    # Add the models to evaluate parameter
    if [[ -n "$MODELS_TO_EVALUATE" ]]; then
        CMD="$CMD --models $MODELS_TO_EVALUATE"
    fi
else
    CMD="python compare_images.py"
    # Add comparison model parameters
    if [[ -n "$MODEL_A" ]]; then
        CMD="$CMD --model-a $MODEL_A"
    fi
    if [[ -n "$MODEL_B" ]]; then
        CMD="$CMD --model-b $MODEL_B"
    fi
    if [[ -n "$MODEL_A_NAME" ]]; then
        CMD="$CMD --model-a-name $MODEL_A_NAME"
    fi
    if [[ -n "$MODEL_B_NAME" ]]; then
        CMD="$CMD --model-b-name $MODEL_B_NAME"
    fi
fi

# Add other parameters
CMD="$CMD --dataset $DATASET $TEST_MODE --sample-size $SAMPLE_SIZE --max-retries $MAX_RETRIES --timeout $TIMEOUT $DEBUG_MODE"

# Show the command to be executed
echo -e "${YELLOW}Executing command:${NC} $CMD"
echo ""

# Ensure the current directory is the script's directory before executing
cd "$SCRIPT_DIR"

# Execute the command
eval $CMD

# Display completion message
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Evaluation Completed!${NC}"
    if [[ "$MODE" == "evaluate" ]]; then
        result_dir="results_evaluate"
    else
        result_dir="results_compare"
    fi
    echo -e "Results saved in ${BLUE}${SCRIPT_DIR}/${result_dir}${NC} directory"
else
    echo -e "\n${RED}Error occurred during evaluation${NC}"
fi
