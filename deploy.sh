#!/bin/bash

# Triton NLP Service Deployment Script

set -e

echo "======================================"
echo "Triton NLP Service Deployment"
echo "======================================"

# Configuration
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"your-registry"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
NAMESPACE=${NAMESPACE:-"triton-nlp"}
DEPLOYMENT_TYPE=${1:-"docker"}  # docker, k8s, or local

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            print_error "Docker is not installed"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            print_error "Docker Compose is not installed"
            exit 1
        fi
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "k8s" ]]; then
        if ! command -v kubectl &> /dev/null; then
            print_error "kubectl is not installed"
            exit 1
        fi
        
        # Check kubectl connection
        if ! kubectl cluster-info &> /dev/null; then
            print_error "kubectl is not connected to a cluster"
            exit 1
        fi
    fi
    
    print_status "Prerequisites check passed"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    docker build -t triton-nlp-service:${IMAGE_TAG} .
    
    if [[ -n "$DOCKER_REGISTRY" && "$DOCKER_REGISTRY" != "your-registry" ]]; then
        docker tag triton-nlp-service:${IMAGE_TAG} ${DOCKER_REGISTRY}/triton-nlp-service:${IMAGE_TAG}
        print_status "Tagged image as ${DOCKER_REGISTRY}/triton-nlp-service:${IMAGE_TAG}"
    fi
    
    print_status "Docker image built successfully"
}

# Push image to registry
push_image() {
    if [[ -n "$DOCKER_REGISTRY" && "$DOCKER_REGISTRY" != "your-registry" ]]; then
        print_status "Pushing image to registry..."
        docker push ${DOCKER_REGISTRY}/triton-nlp-service:${IMAGE_TAG}
        print_status "Image pushed successfully"
    else
        print_warning "DOCKER_REGISTRY not set, skipping push"
    fi
}

# Deploy using Docker Compose
deploy_docker() {
    print_status "Deploying with Docker Compose..."
    
    # Stop existing containers
    docker-compose down || true
    
    # Start services
    docker-compose up -d
    
    # Wait for service to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/v2/health/ready &> /dev/null; then
        print_status "Triton server is ready"
    else
        print_error "Triton server health check failed"
        docker-compose logs triton-nlp
        exit 1
    fi
    
    print_status "Docker deployment completed successfully"
    print_status "Services available at:"
    echo "  - HTTP: http://localhost:8000"
    echo "  - gRPC: localhost:8001"
    echo "  - Metrics: http://localhost:8002"
    echo "  - FastAPI: http://localhost:8080"
}

# Deploy to Kubernetes
deploy_k8s() {
    print_status "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image in deployment
    if [[ -n "$DOCKER_REGISTRY" && "$DOCKER_REGISTRY" != "your-registry" ]]; then
        sed -i "s|image: triton-nlp-service:latest|image: ${DOCKER_REGISTRY}/triton-nlp-service:${IMAGE_TAG}|g" deployment/k8s-deployment.yaml
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/k8s-deployment.yaml -n ${NAMESPACE}
    
    # Wait for deployment to be ready
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/triton-nlp-server -n ${NAMESPACE}
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get svc triton-nlp-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    print_status "Kubernetes deployment completed successfully"
    print_status "Deployment status:"
    kubectl get pods -n ${NAMESPACE}
    
    if [[ "$SERVICE_IP" != "pending" ]]; then
        print_status "Services available at:"
        echo "  - HTTP: http://${SERVICE_IP}:8000"
        echo "  - gRPC: ${SERVICE_IP}:8001"
        echo "  - Metrics: http://${SERVICE_IP}:8002"
    else
        print_warning "LoadBalancer IP is pending. Use kubectl port-forward for local access:"
        echo "  kubectl port-forward -n ${NAMESPACE} svc/triton-nlp-service 8000:8000 8001:8001"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    # Install client dependencies
    pip3 install tritonclient[all] numpy
    
    # Run test client
    python3 client/triton_client.py --test
    
    print_status "Tests completed"
}

# Clean up
cleanup() {
    print_status "Cleaning up..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        docker-compose down
        print_status "Docker containers stopped"
    elif [[ "$DEPLOYMENT_TYPE" == "k8s" ]]; then
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        print_status "Kubernetes namespace deleted"
    fi
}

# Main deployment flow
main() {
    case "$1" in
        build)
            check_prerequisites
            build_image
            ;;
        push)
            check_prerequisites
            push_image
            ;;
        docker)
            check_prerequisites
            build_image
            deploy_docker
            ;;
        k8s|kubernetes)
            check_prerequisites
            build_image
            push_image
            deploy_k8s
            ;;
        test)
            run_tests
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo "Usage: $0 {build|push|docker|k8s|test|cleanup}"
            echo ""
            echo "Commands:"
            echo "  build    - Build Docker image"
            echo "  push     - Push image to registry"
            echo "  docker   - Deploy using Docker Compose"
            echo "  k8s      - Deploy to Kubernetes"
            echo "  test     - Run tests"
            echo "  cleanup  - Clean up deployment"
            echo ""
            echo "Environment variables:"
            echo "  DOCKER_REGISTRY - Docker registry URL"
            echo "  IMAGE_TAG       - Image tag (default: latest)"
            echo "  NAMESPACE       - Kubernetes namespace (default: triton-nlp)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
