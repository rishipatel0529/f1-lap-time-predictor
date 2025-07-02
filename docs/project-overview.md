## Objectives & Success Metrics

### Business & Race Outcomes
- **Average lap-time reduction**: ≥ 0.3 s  
- **Mean Time-To-Pit (MTTP) improvement**: ≥ 1.5 s  
- **Pit-stop efficiency**: ≤ 2.5 s from entry to exit (end-to-end pit service)  

### Model Accuracy & Learning
- **Tire-wear forecast RMSE**: ≤ 0.2 °C  
- **Pit-delta prediction error**: MAE ≤ 0.5 s  
- **RL policy reward uplift** over rule-based baseline: ≥ 10 % average reward  

### Performance & Scalability
- **End-to-end prediction latency** (telemetry → response): P95 ≤ 100 ms  
- **Model inference throughput**: ≥ 2000 requests/sec without degradation  
- **Feature store read latency**: P95 ≤ 30 ms  
- **Kafka ingestion throughput**: ≥ 5000 events/sec sustained  

### Reliability & Operations
- **System availability**: ≥ 99.9 % uptime  
- **Pipeline durability** (message loss rate): ≤ 0.5 %  
- **Full retraining pipeline run-time**: ≤ 4 hours  
- **Alert delivery latency** (model/data-drift or errors): ≤ 1 minute  

### Quality & Maintainability
- **Unit-test coverage**: ≥ 85 % across `src/`  
- **CI green build rate**: ≥ 95 % on `main` branch  
- **Documentation completeness**: all new modules ship with README/examples  

