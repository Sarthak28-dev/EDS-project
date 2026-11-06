# RouteWise Data Dictionary
## Comprehensive Variable Specifications for Exploratory Data Science Evaluators

This data dictionary provides full metadata, structural definitions, validation rules, and statistical distributions for all attributes in the RouteWise delivery tracking system.

---

### Variable Specifications

| Field Name | Data Type | Nullable | Domain / Allowed Values | Units | Description & Operational Significance |
|---|---|---|---|---|---|
| `delivery_id` | `string` | No | Format `DEL-[0-9]{5}` | N/A | Unique shipment identifier generated upon warehouse order routing. Primary key for shipment auditing. |
| `order_date` | `date` | No | `2024-01-01` to `2024-06-30` | Date | Date when customer transaction was confirmed and order queued in the ERP system. |
| `dispatch_time` | `datetime` | No | `YYYY-MM-DD HH:MM:SS` | Timestamp | Exact date and time when the transport vehicle cleared the origin warehouse gate. |
| `delivery_time` | `datetime` | No | `YYYY-MM-DD HH:MM:SS` | Timestamp | Exact date and time when the customer signed for delivery at destination. Strictly greater than `dispatch_time`. |
| `route` | `string` (nominal) | No | `Route_Central`, `Route_North`, `Route_South`, `Route_East`, `Route_West` | N/A | Designated commercial delivery lane connecting specific origins and destination hubs. |
| `origin` | `string` (nominal) | No | `Warehouse_Central`, `Warehouse_North`, `Warehouse_South`, `Warehouse_West` | N/A | Regional dispatch depot where consignments are consolidated and loaded. |
| `destination` | `string` (nominal) | No | `Zone_Metro`, `Zone_Suburbs`, `Zone_Industrial`, `Zone_Rural` | N/A | Geographic classification of customer delivery point. |
| `carrier` | `string` (nominal) | No | `ApexDelivery`, `FastShip`, `NationalExpress`, `QuickLogistics` | N/A | 3PL (third-party logistics) contracted carrier responsible for transport and last-mile delivery. |
| `weather` | `string` (nominal) | No | `Clear`, `Rainy`, `Foggy`, `Snowy`, `Stormy` | N/A | Prevailing weather condition recorded across transit corridor during the delivery window. |
| `parcel_weight_kg` | `float64` (ratio) | No | Range: `0.1` to `40.0` | Kilograms (kg) | Total gross weight of the parcel package as weighed on warehouse calibrated scales. |
| `distance_km` | `float64` (ratio) | No | Range: `5.0` to `450.0` | Kilometres (km) | Actual transit odometer distance between warehouse origin and recipient drop-off. |
| `traffic_level` | `string` (ordinal) | No | `Low`, `Medium`, `High`, `Jam` | Intensity Level | Road traffic density recorded via telematics. Natural hierarchy: $\text{Low} < \text{Medium} < \text{High} < \text{Jam}$. |
| `delivery_time_hours` | `float64` (continuous) | No | Range: `0.4` to `20.0` | Decimal Hours | Total transit duration from warehouse dispatch to recipient delivery. Target variable for regression models. |
| `delayed` | `int64` (binary) | No | `0` (On-time), `1` (Delayed) | Binary Indicator | Boolean flag denoting whether total delivery duration exceeded SLA threshold buffer for that corridor. |

---

### Variable Categorization by Role in Analysis & Modelling

```
+--------------------------------------------------------------------------------+
|                             RouteWise Variables                                |
+--------------------------------------------------------------------------------+
          |                                               |
  [Descriptive / Identifiers]                    [Analytical Variables]
  - delivery_id                                           |
  - order_date                               +------------+------------+
  - dispatch_time                            |                         |
  - delivery_time (Audit only)        [Predictors (X)]          [Targets (y)]
                                      - distance_km             - delivery_time_hours
                                      - parcel_weight_kg          (Continuous)
                                      - route                   - delayed
                                      - carrier                   (Binary Audit)
                                      - weather
                                      - traffic_level
                                      - origin
                                      - destination
```

---

### Audit & Cleaning Rules Applied

1. **Identifier Uniqueness**: All records must possess a unique `delivery_id`. Exact and conflicting duplicate rows must be deduplicated.
2. **Temporal Consistency**: `delivery_time` must be chronologically after `dispatch_time`. Inverted timestamps or negative/zero durations represent data acquisition faults and must be eliminated.
3. **Weight Validity**: Weights must be strictly positive and within realistic physical bounds ($w \le 40$ kg for standard parcel courier contracts). Negative signs (entry error) and values scaled by $100\times$ (gram misentry) are normalized.
4. **Canonical Categorical Representations**: Typographical variants (e.g. `fastship`, `Apex_Delivery`, `Natl_Express`) are mapped to canonical names. Erroneous categories (`UNKNOWN`, `med`, `Thunderstorm`) are rectified.
5. **Missing Value Imputation**: Missing values in `weather` and `parcel_weight_kg` must be transparently imputed using domain-consistent methods (median weight by route, mode weather).
