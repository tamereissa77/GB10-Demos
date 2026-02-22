# VSS Agent Management Makefile
#
# Usage:
#   make start-all             - Start all services (Event Reviewer + CV Detector)
#   make stop-all              - Stop all services
#   make restart-all           - Restart all services
#   make logs-event            - View logs for Event Reviewer (via-server)
#   make logs-cv               - View logs for CV Event Detector
#   make clean                 - Stop and remove all containers and networks
#
# Individual Service Management:
#   make start-event           - Start VSS Event Reviewer only
#   make start-cv              - Start CV Event Detector only
#   make stop-event            - Stop VSS Event Reviewer
#   make stop-cv               - Stop CV Event Detector

# Environment Configuration
BASE_DIR := $(shell pwd)
EVENT_REVIEWER_DIR := deploy/docker/event_reviewer
CV_DETECTOR_DIR := examples/cv-event-detector
ENV_VARS := IS_SBSA=1 ALERT_REVIEW_MEDIA_BASE_DIR=/tmp/alert-media-dir

.PHONY: help start-all stop-all restart-all logs-event logs-cv clean start-event start-cv stop-event stop-cv

help:
	@echo "VSS Agent Management Makefile"
	@echo "-----------------------------"
	@echo "Usage:"
	@echo "  make start-all             - Start all services (Event Reviewer + CV Detector)"
	@echo "  make stop-all              - Stop all services"
	@echo "  make restart-all           - Restart all services"
	@echo "  make status                - Check status of containers"
	@echo "  make logs-event            - View logs for Event Reviewer (via-server)"
	@echo "  make logs-cv               - View logs for CV Event Detector"
	@echo "  make clean                 - Stop and remove all containers and networks"
	@echo ""
	@echo "  make start-event           - Start VSS Event Reviewer only"
	@echo "  make start-cv              - Start CV Event Detector only"
	@echo "  make stop-event            - Stop VSS Event Reviewer"
	@echo "  make stop-cv               - Stop CV Event Detector"
	@echo "  make cache-cleaner         - Run system cache cleaner (background)"

start-all: start-event start-cv
	@echo ""
	@echo "All services started successfully! :rocket:"
	@echo ""
	@echo "Access your VSS Agent here:"
	@echo "---------------------------------------------------------"
	@echo "Alert Inspector UI:    http://localhost:7860"
	@echo "CV Event Detector UI:  http://localhost:7862"
	@echo "VSS Backend Docs:      http://localhost:8000/docs"
	@echo "API Gateway:           http://localhost:8080"
	@echo "---------------------------------------------------------"
	@echo ""

stop-all: stop-event stop-cv
	@echo ""
	@echo "All services are down."
	@echo ""

restart-all: stop-all start-all

status:
	@docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"

start-event:
	@echo "Starting VSS Event Reviewer..."
	@cd $(EVENT_REVIEWER_DIR) && $(ENV_VARS) docker compose up -d

stop-event:
	@echo "Stopping VSS Event Reviewer..."
	@cd $(EVENT_REVIEWER_DIR) && $(ENV_VARS) docker compose down

start-cv:
	@echo "Starting CV Event Detector..."
	@cd $(CV_DETECTOR_DIR) && $(ENV_VARS) docker compose up -d

stop-cv:
	@echo "Stopping CV Event Detector..."
	@cd $(CV_DETECTOR_DIR) && $(ENV_VARS) docker compose down

logs-event:
	@docker logs -f event_reviewer-via-server-1

logs-cv:
	@cd $(CV_DETECTOR_DIR) && docker compose logs -f

clean: stop-all
	@echo "Removing Docker networks..."
	@docker network prune -f
	@echo "Cleanup complete."

cache-cleaner:
	@echo "Starting system cache cleaner..."
	@sudo sh deploy/scripts/sys_cache_cleaner.sh &
