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


## Data Sources

- **Live telemetry**  
  - **Topics**  
    - `f1-telemetry-packets`: Raw UDP packets, parsed downstream  
    - `f1-telemetry-drivers`: Per‐driver state (position, lap/sector times, DRS, etc.)  
    - `f1-telemetry-events`: Race events (fastest lap, speed trap, safety car, etc.)  
  - **Fields consumed**  
    - **Performance**: `speed`, `engine_rpm`, `gear`, `throttle_pct`, `brake_pressure`, wheel speeds  
    - **Tire**: `tire_temp_FL/FR/RL/RR`, `tire_pressure_FL/FR/RL/RR`  
    - **Car status**: `fuel_level`, `engine_temp`, `brake_temp`, `suspension_travel`, `wing_damage`  
    - **Acceleration**: `g_lat`, `g_long`  
    - **Driver/Race info**: `lap_time`, `sector_time`, `position`, `drs_active`  
  - **Ingestion**: Kafka cluster at `kafka:9092`, topics as above

- **Weather API**  
  - **Provider**: OpenWeatherMap One Call API 3.0  
  - **Endpoint**:  
    ```  
    GET https://api.openweathermap.org/data/3.0/onecall
        ?lat={lat}&lon={lon}&units=metric&appid=${OWM_API_KEY}
    ```  
  - **Fields consumed** (JSON → feature name):  
    - `current.temp` → temperature (°C)  
    - `current.humidity` → humidity (%)  
    - `current.wind_speed` → wind_speed (m/s)  
    - `current.wind_gust` → wind_gust (m/s)  
    - `current.feels_like` → feels_like_temp (°C)  
    - `minutely[].precipitation` → precipitation_mm (take max per minute)  
    - `hourly[0].pop` → precip_probability (0–1)  
  - **Access**: API key in `config/.env` as `OWM_API_KEY`

- **Historical race logs**  
  - **Source**:  
    - Ergast API (World Championship data, 1973–present)  
    - FastF1 telemetry dumps (2018–present)  
  - **Coverage**: 46+ years of F1 data for back-testing and training  
  - **Storage**: Parquet files under `data/historical/{season}/{race}/…`  
  - **Fields consumed**:  
    - `lap_time`, `sector_times`, `pit_stop_duration`, `compound`  
    - `grid_position`, `finish_position`, `weather_condition`

## Stakeholders & Users

- **F1 strategy engineers**: will consume real‐time recommendations  
- **Data science team**: will develop models and review metrics  
- **Race operations dashboard users**: will view insights during live events  


## Data Schema Draft

### Live Telemetry Stream
| Field                      | Type     | Notes                                               |
|----------------------------|----------|-----------------------------------------------------|
| car_id                     | string   | Unique car identifier (e.g. “MER01”)                |
| driver_id                  | string   | Unique driver code                                  |
| timestamp                  | datetime | ISO-8601 or Unix epoch of the telemetry event       |
| speed                      | float    | km/h                                                |
| engine_rpm                 | int      | revolutions per minute                              |
| gear                       | int      | current gear (1–8)                                  |
| throttle_pct               | float    | % throttle application                              |
| brake_pressure             | float    | bar                                                 |
| wheel_speed_FL/FR/RL/RR    | float    | km/h, individual wheel speeds                       |
| tire_temp_FL/FR/RL/RR      | float    | °C, individual tire temperatures                    |
| tire_pressure_FL/FR/RL/RR  | float    | PSI, individual tire pressures                      |
| fuel_level                 | float    | liters remaining                                    |
| engine_temp                | float    | °C                                                  |
| brake_temp                 | float    | °C                                                  |
| suspension_travel          | float    | mm or m (specify)                                   |
| wing_damage                | bool     | damage flag                                         |
| g_lat/g_long               | float    | lateral & longitudinal acceleration (g)             |
| lap_time                   | float    | s, last completed lap                               |
| sector_time                | float    | s, last completed sector                            |
| position                   | int      | current track position                              |
| drs_active                 | bool     | DRS on/off                                          |

### Weather API Response
| Field                           | Type       | Notes                                        |
|---------------------------------|------------|----------------------------------------------|
| lat, lon                        | float      | query coordinates                            |
| timezone, timezone_offset       | string/int | timezone name & offset (s)                   |
| current.temp                    | float      | °C                                           |
| current.feels_like              | float      | °C                                           |
| current.humidity                | int        | %                                            |
| current.pressure                | int        | hPa                                          |
| current.visibility              | int        | meters                                       |
| current.clouds                  | int        | % cloud cover                                |
| current.wind_speed              | float      | m/s                                          |
| current.wind_gust               | float      | m/s                                          |
| current.wind_deg                | int        | wind direction (°)                           |
| minutely[].precipitation        | float      | mm/min, take max                             |
| hourly[0].temp                  | float      | °C, 1-h forecast                             |
| hourly[0].pop                   | float      | 0–1, precip probability in next hour         |
| hourly[0].weather[0].main       | string     | forecast condition (e.g. “Rain”)             |
| daily[0].temp.max / temp.min    | float      | °C, day-ahead high/low                       |
| daily[0].pop                    | float      | 0–1, day-ahead precip probability            |

### Historical Race Logs
| Field               | Type     | Notes                                                      |
|---------------------|----------|------------------------------------------------------------|
| season              | int      | championship year                                          |
| race_name           | string   | Grand Prix name                                            |
| race_date           | date     | race calendar date                                         |
| session_type        | string   | Practice / Qualifying / Race                               |
| lap_number          | int      | lap index                                                  |
| driver              | string   | driver code or ID                                          |
| car_number          | int      | entry number                                               |
| lap_time            | float    | s                                                          |
| sector1_time        | float    | s                                                          |
| sector2_time        | float    | s                                                          |
| sector3_time        | float    | s                                                          |
| pit_stop_lap        | int      | lap during which pit stop occurred                         |
| pit_stop_duration   | float    | s                                                          |
| compound            | string   | tire compound (Soft/Medium/Hard/Wet/Intermediate)          |
| grid_position       | int      | start position                                             |
| finish_position     | int      | race finish position                                       |
| weather_condition   | string   | e.g. “Sunny”, “Light rain”                                 |
| ambient_temp        | float    | °C at race start                                           |
| track_temp          | float    | °C surface temp (if available)                             |
| track_condition     | string   | e.g. “Dry”, “Wet”                                          |


## Tech Stack Decisions

### Kafka (self-managed) vs. Confluent Cloud
- **Self-managed Kafka**  
  - **Pros:** Full operational control, no vendor lock-in, lower per-message cost at scale  
  - **Cons:** Significant DevOps overhead (cluster setup, upgrades, scaling, monitoring)  
- **Confluent Cloud**  
  - **Pros:** Fully managed service (zero-ops), enterprise SLAs, built-in connectors & schema registry  
  - **Cons:** Higher unit cost, potential vendor lock-in  

### Feast OSS vs. SageMaker Feature Store
- **Feast OSS**  
  - **Pros:** Open-source, flexible backends (Redis, BigQuery, etc.), active community, easy local dev  
  - **Cons:** You manage hosting, scaling, upgrades  
- **SageMaker Feature Store**  
  - **Pros:** Fully managed on AWS, integrated with SageMaker Studio & IAM, built-in online/offline stores  
  - **Cons:** AWS-only, higher cost, less flexibility in storage engines  

### Airflow vs. Prefect
- **Apache Airflow**  
  - **Pros:** Battle-tested, large ecosystem of operators, mature UI for DAG management  
  - **Cons:** Can be heavyweight to configure, YAML/python provenance more verbose  
- **Prefect**  
  - **Pros:** Python-native DAGs, simpler API, dynamic task mapping, built-in retries & state handling  
  - **Cons:** Younger ecosystem, fewer community plugins (but rapidly growing)  

### Minikube vs. EKS
- **Minikube**  
  - **Pros:** Zero-cost local Kubernetes cluster for dev/debug, fast iteration  
  - **Cons:** Not production-grade, limited cluster resources  
- **Amazon EKS**  
  - **Pros:** Fully managed Kubernetes in AWS, auto-scaling nodes, integrated IAM & networking  
  - **Cons:** More complex to set up, higher operational cost  

---

**Selections for this project**  
- **Streaming:** Self-managed Kafka on Kubernetes for dev/test; evaluate Confluent Cloud for production roll-out  
- **Feature Store:** Feast OSS (open-source, easy local integration)  
- **Orchestration:** Apache Airflow (maturity & ecosystem align with our CI/CD pipelines)  
- **Kubernetes:** Minikube for local development, EKS for production deployments  
