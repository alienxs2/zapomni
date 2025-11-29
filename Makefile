.PHONY: test e2e docker-up docker-down server lint clean help load-test load-test-ui load-test-light

# Variables
PYTHON := python3
PYTEST := $(PYTHON) -m pytest
HOST := 127.0.0.1
PORT := 8000

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

test: ## Run unit tests
	$(PYTEST) tests/unit/ -q

e2e: ## Run E2E tests (requires running server)
	$(PYTEST) tests/e2e/ -v

docker-up: ## Start Docker services (FalkorDB + Redis)
	docker compose up -d

docker-down: ## Stop Docker services
	docker compose down

server: ## Start MCP server
	$(PYTHON) -m zapomni_mcp --host $(HOST) --port $(PORT)

lint: ## Run linters (black, isort)
	black --check src/ tests/
	isort --check src/ tests/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

clean: ## Clean cache and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov

coverage: ## Run tests with coverage
	$(PYTEST) tests/unit/ --cov=src --cov-report=html --cov-report=term

all: lint test ## Run lint and tests

# Load Testing
load-test: ## Run load test (50 users, 5 min)
	@echo "Starting load test (50 users, 5 min)..."
	locust -f tests/load/locustfile.py --host=http://$(HOST):$(PORT) \
		--headless -u 50 -r 10 --run-time 5m

load-test-ui: ## Run Locust with Web UI at http://localhost:8089
	@echo "Starting Locust Web UI at http://localhost:8089..."
	locust -f tests/load/locustfile.py --host=http://$(HOST):$(PORT)

load-test-light: ## Light load test (10 users, 1 min)
	@echo "Light load test (10 users, 1 min)..."
	locust -f tests/load/locustfile.py --host=http://$(HOST):$(PORT) \
		--headless -u 10 -r 5 --run-time 1m
