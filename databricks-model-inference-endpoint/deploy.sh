#!/bin/bash

# Databricks Model Inference Endpoint Deployment Script
# This script provides an easy way to deploy the Political Party Classification model endpoint

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Python is installed
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command_exists pip3; then
        print_error "pip3 is not installed"
        exit 1
    fi
    
    # Check if Databricks CLI is configured
    if ! command_exists databricks; then
        print_warning "Databricks CLI not found. Installing..."
        pip3 install databricks-cli
    fi
    
    # Check if databricks is configured
    if ! databricks fs ls >/dev/null 2>&1; then
        print_error "Databricks CLI not configured. Please run 'databricks configure' first"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Function to deploy endpoint
deploy_endpoint() {
    local config_file=$1
    local verbose=$2
    
    print_status "Deploying model inference endpoint..."
    
    # Build command
    cmd="python3 deploy_endpoint.py"
    
    if [ -n "$config_file" ]; then
        cmd="$cmd --config-file $config_file"
    fi
    
    if [ "$verbose" = "true" ]; then
        cmd="$cmd --verbose"
    fi
    
    # Execute deployment
    print_status "Running: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        print_success "Endpoint deployed successfully!"
    else
        print_error "Endpoint deployment failed"
        exit 1
    fi
}

# Function to test endpoint
test_endpoint() {
    local test_mode=$1
    local verbose=$2
    
    print_status "Testing deployed endpoint..."
    
    # Check if deployment info exists
    if [ ! -f "deployment_info.json" ]; then
        print_error "deployment_info.json not found. Please deploy the endpoint first."
        exit 1
    fi
    
    # Build command
    cmd="python3 test_endpoint.py --test-mode $test_mode"
    
    if [ "$verbose" = "true" ]; then
        cmd="$cmd --verbose"
    fi
    
    # Execute test
    print_status "Running: $cmd"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        print_success "Endpoint test passed!"
    else
        print_error "Endpoint test failed"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  deploy [CONFIG_FILE]  Deploy the model inference endpoint"
    echo "  test [MODE]          Test the deployed endpoint"
    echo "  full                 Deploy and test the endpoint"
    echo "  help                 Show this help message"
    echo ""
    echo "Options:"
    echo "  --verbose, -v        Enable verbose output"
    echo "  --config FILE        Configuration file (default: config.json)"
    echo ""
    echo "Test Modes:"
    echo "  quick                Quick test (health check + single prediction)"
    echo "  full                 Full test suite (default)"
    echo "  performance          Performance test only"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                    # Deploy with default config"
    echo "  $0 deploy custom_config.json # Deploy with custom config"
    echo "  $0 test quick                # Quick test"
    echo "  $0 test full --verbose       # Full test with verbose output"
    echo "  $0 full                      # Deploy and test"
}

# Function to show UI deployment instructions
show_ui_instructions() {
    echo ""
    echo "=========================================="
    echo "UI DEPLOYMENT INSTRUCTIONS"
    echo "=========================================="
    echo ""
    echo "If you prefer to deploy using the Databricks UI:"
    echo ""
    echo "1. Open your Databricks workspace"
    echo "2. Navigate to Machine Learning > Serving Endpoints"
    echo "3. Click 'Create Serving Endpoint'"
    echo "4. Enter the following details:"
    echo "   - Name: political-party-classifier-endpoint"
    echo "   - Description: Political Party Classification Model"
    echo "5. Click 'Add Model' and select:"
    echo "   - Model Source: Unity Catalog"
    echo "   - Model: mle_batch_catalog_2025_q2.mle_shyamkumar_vn.political_party_classifier"
    echo "   - Version: Latest"
    echo "   - Workload Size: Small"
    echo "6. Enable 'Scale to Zero'"
    echo "7. Add environment variables:"
    echo "   - CATALOG_NAME=mle_batch_catalog_2025_q2"
    echo "   - SCHEMA_NAME=mle_shyamkumar_vn"
    echo "   - MODEL_NAME=political_party_classifier"
    echo "   - PRODUCTION_ALIAS=production"
    echo "8. Click 'Create Endpoint'"
    echo ""
    echo "For detailed instructions, see README.md"
    echo ""
}

# Main script logic
main() {
    # Parse arguments
    COMMAND=""
    CONFIG_FILE="config.json"
    TEST_MODE="full"
    VERBOSE="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            deploy|test|full|help)
                COMMAND="$1"
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --verbose|-v)
                VERBOSE="true"
                shift
                ;;
            quick|full|performance)
                if [ "$COMMAND" = "test" ]; then
                    TEST_MODE="$1"
                fi
                shift
                ;;
            *)
                if [ -z "$COMMAND" ]; then
                    COMMAND="$1"
                elif [ "$COMMAND" = "deploy" ] && [ -f "$1" ]; then
                    CONFIG_FILE="$1"
                fi
                shift
                ;;
        esac
    done
    
    # Show help if no command or help requested
    if [ -z "$COMMAND" ] || [ "$COMMAND" = "help" ]; then
        show_usage
        show_ui_instructions
        exit 0
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Install dependencies
    install_dependencies
    
    # Execute command
    case $COMMAND in
        deploy)
            deploy_endpoint "$CONFIG_FILE" "$VERBOSE"
            ;;
        test)
            test_endpoint "$TEST_MODE" "$VERBOSE"
            ;;
        full)
            deploy_endpoint "$CONFIG_FILE" "$VERBOSE"
            test_endpoint "$TEST_MODE" "$VERBOSE"
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
    
    print_success "Operation completed successfully!"
}

# Run main function
main "$@" 