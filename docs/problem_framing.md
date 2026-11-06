# RouteWise: Supply Chain Delivery Analytics & Prediction
## Problem Framing & Operational Context

---

### 1. Business Scenario
A nationwide fictional distributor, **RouteWise Logistics Network**, orchestrates delivery operations across five critical commercial transit corridors connecting regional distribution centers (warehouses) to metro, suburban, industrial, and rural fulfillment clusters. In modern logistics, supply chains face multi-dimensional friction: unpredictable weather fronts, road traffic congestion, carrier scheduling variability, and differing cargo handling requirements based on consignment weights. 

RouteWise currently experiences fluctuating delivery times that degrade customer satisfaction, create downstream warehouse bottlenecks, and trigger SLA penalty costs.

---

### 2. Business Problem
Delivery lead times frequently exceed promised Service Level Agreements (SLAs), yet operations teams lack quantifiable visibility into the root drivers of these delays. Management currently relies on anecdotal explanations—often attributing delays exclusively to external bad weather or carrier underperformance—without empirical verification. The company urgently requires an evidence-based analytics foundation to determine the relative impact of carrier operational efficiency, route topography/distance, road traffic congestion, parcel weight, and atmospheric conditions on delivery durations.

---

### 3. Key Stakeholders

| Stakeholder Group | Operational Focus | Primary Needs & Interests |
|---|---|---|
| **VP of Supply Chain Operations** | Network-wide efficiency, SLA compliance, cost control | Macro performance visibility, carrier SLA adherence rates, delay bottleneck identification. |
| **Logistics Dispatch Managers** | Daily operational scheduling & route dispatching | Actionable thresholds for rerouting or adjusting dispatch times under adverse weather/traffic. |
| **Carrier Procurement & Partner Relations** | Contract negotiation, carrier vendor management | Empirical benchmarks comparing carrier velocity, reliability, and route-specific efficiencies. |
| **Customer Experience & Sales Teams** | Accurate delivery promises, client trust | Realistic estimated times of arrival (ETAs) to prevent customer churn. |
| **Data Science & ML Engineering (Person 2)** | Predictive model development | Clean, high-integrity dataset free of target leakage, ready for regression pipelines. |

---

### 4. Strategic & Operational Decisions Supported
1. **Dynamic Carrier Allocation**: Route parcels dynamically to carriers with the highest historical velocity and reliability on specific route types.
2. **Buffer Adjustment in SLAs**: Re-baseline promised delivery windows according to seasonal weather patterns and real-time traffic levels rather than flat, static buffers.
3. **Route Rerouting & Optimization**: Re-evaluate routes that consistently suffer from structural congestion or disproportionate delay percentages.
4. **Heavy Parcel Surcharges & Specialized Fleet Routing**: Determine if parcel weight imposes substantial handling delays that warrant specialized handling protocols.

---

### 5. Analytical Questions

#### Primary Analytical Question
> **"How accurately can delivery time be predicted using route, carrier, weather, parcel weight, distance, and traffic conditions?"**

#### Supporting Analytical Questions
1. **Weather Impact**: Do adverse atmospheric conditions (rain, fog, snow, storms) cause statistically significant increases in delivery duration compared to clear weather?
2. **Carrier Disparity**: Are observed differences in delivery speed across the four carriers statistically meaningful, or do they simply reflect differences in the routes they are assigned?
3. **Traffic Bottlenecks**: How non-linear is the delay penalty between "Medium", "High", and "Jam" traffic conditions across urban vs. intercity corridors?
4. **Parcel Weight Sensitivity**: Does parcel weight demonstrate a strong physical correlation with delivery duration, or is handling time negligible relative to transit distance?
5. **Route-Specific Delay Profiles**: Which specific origin-destination corridors exhibit the highest proportion of SLA violations?

---

### 6. Dataset Provenance & Synthetic Disclaimer
> **Academic EDS Disclaimer**:  
> The dataset utilized in this study (`data/raw/delivery_data.csv` and `data/processed/cleaned_delivery_data.csv`) is a synthetically generated dataset engineered for the **RouteWise Exploratory Data Science** project. While parameterized using authentic freight physics, empirical traffic multipliers, and stochastic noise to simulate realistic distributor operations, the records do not represent any real-world commercial entity or proprietary customer data.

---

### 7. Variable Taxonomy & Roles

| Variable | Type | Role in Pipeline | Measurement Unit |
|---|---|---|---|
| `delivery_id` | Identifier | Record tracking key | Alpha-numeric (`DEL-XXXXX`) |
| `order_date` | Temporal | Temporal index / Seasonality | Date (`YYYY-MM-DD`) |
| `dispatch_time` | Temporal | Operational timestamp | Timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `delivery_time` | Temporal | Audit timestamp (Target leakage in ML) | Timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `route` | Categorical | Primary spatial predictor | Nominal code |
| `origin` | Categorical | Logistics node predictor | Nominal code |
| `destination` | Categorical | Destination zone predictor | Nominal code |
| `carrier` | Categorical | Vendor predictor | Nominal code |
| `weather` | Categorical | Environmental predictor | Nominal categories |
| `parcel_weight_kg` | Numerical (Ratio) | Physical cargo predictor | Kilograms (kg) |
| `distance_km` | Numerical (Ratio) | Primary physical predictor | Kilometres (km) |
| `traffic_level` | Categorical (Ordinal) | Roadway friction predictor | Low < Medium < High < Jam |
| `delivery_time_hours` | Numerical (Continuous) | **Primary Target Variable ($y$)** | Decimal Hours (hrs) |
| `delayed` | Categorical (Binary) | Operational SLA audit target | Indicator (0 = On-time, 1 = Delayed) |

---

### 8. Expected Outcomes
- A thoroughly audited, validated, and reproducible data pipeline.
- Identification of at least 3 genuine operational patterns directly informing distributor logistics.
- Statistical hypothesis testing establishing empirical significance of environmental factors.
- Seamless handoff of `cleaned_delivery_data.csv` and `data_contract.md` to Person 2 for linear and non-linear regression modelling.
