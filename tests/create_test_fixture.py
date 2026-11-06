"""
RouteWise: Test Data Fixture Generator

This script generates a realistic temporary test fixture strictly following the
agreed 14-column dataset contract for testing Person 2's modeling and regression
pipelines prior to Person 1's final cleaned delivery dataset delivery.
"""

import csv
import datetime
import math
import os
import random

def generate_fixture(
    num_rows: int = 1200,
    output_path: str = "data/processed/temp_delivery_data_fixture.csv",
    seed: int = 42
) -> str:
    random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    routes = [
        ("Route_A_NorthSouth", "Depot North", "Hub South", 1.0),
        ("Route_B_UrbanExpress", "Metro West", "Metro East", 1.08),
        ("Route_C_Coastway", "Harbor Port", "Coastal City", 1.02),
        ("Route_D_Mountain", "Highland Base", "Valley Center", 1.22),
        ("Route_E_MidlandLink", "Central Depot", "East Distribution", 1.0),
    ]

    carriers = [
        ("SwiftLog", 0.96),
        ("ApexDelivery", 1.00),
        ("MetroFreight", 1.04),
        ("EcoHaul", 1.10),
    ]

    weather_conditions = [
        ("Clear", 1.00, 0.45),
        ("Rainy", 1.15, 0.25),
        ("Foggy", 1.20, 0.12),
        ("Snowy", 1.45, 0.10),
        ("Stormy", 1.70, 0.08),
    ]

    traffic_levels = [
        ("Low", 1.00, 0.35),
        ("Medium", 1.25, 0.35),
        ("High", 1.60, 0.20),
        ("Severe", 2.10, 0.10),
    ]

    weather_choices, weather_factors, weather_weights = zip(*weather_conditions)
    traffic_choices, traffic_factors, traffic_weights = zip(*traffic_levels)

    start_date = datetime.date(2024, 1, 1)

    fieldnames = [
        "delivery_id",
        "order_date",
        "dispatch_time",
        "delivery_time",
        "route",
        "origin",
        "destination",
        "carrier",
        "weather",
        "parcel_weight_kg",
        "distance_km",
        "traffic_level",
        "delivery_time_hours",
        "delayed"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, num_rows + 1):
            delivery_id = f"DEL-{10000 + i}"
            day_offset = random.randint(0, 180)
            order_date = start_date + datetime.timedelta(days=day_offset)

            dispatch_hour = random.randint(6, 17)
            dispatch_min = random.randint(0, 59)
            dispatch_time = datetime.datetime.combine(
                order_date, datetime.time(dispatch_hour, dispatch_min)
            )

            route_name, origin, dest, route_factor = random.choice(routes)
            carrier_name, carrier_factor = random.choice(carriers)

            # Sample weather and traffic based on probabilities
            weather_idx = random.choices(range(len(weather_choices)), weights=weather_weights)[0]
            weather_name = weather_choices[weather_idx]
            weather_mult = weather_factors[weather_idx]

            traffic_idx = random.choices(range(len(traffic_choices)), weights=traffic_weights)[0]
            traffic_name = traffic_choices[traffic_idx]
            traffic_mult = traffic_factors[traffic_idx]

            # Realistic distance: between 15.0 and 420.0 km
            distance_km = round(random.uniform(15.0, 420.0), 1)

            # Realistic parcel weight: between 0.5 and 45.0 kg
            parcel_weight_kg = round(random.uniform(0.5, 45.0), 1)

            # Physics / operational duration model:
            # Baseline transit speed: ~60 km/h
            base_transit_hours = distance_km / 60.0

            # Weight handling penalty: 0.015 hrs per kg
            handling_hours = parcel_weight_kg * 0.015

            # Combined duration with realistic interaction effects
            multiplier = route_factor * carrier_factor * weather_mult * traffic_mult

            # Gaussian noise plus occasional long-tail operational delays (heteroscedasticity)
            noise_std = 0.15 + (distance_km / 400.0) * 0.25
            noise = random.gauss(0, noise_std)

            # 4% chance of acute supply chain shock / mechanical delay (tail outlier)
            shock = random.uniform(1.2, 3.5) if random.random() < 0.04 else 0.0

            raw_duration = (base_transit_hours * multiplier) + handling_hours + noise + shock
            delivery_time_hours = max(0.4, round(raw_duration, 2))

            # Expected SLA duration: distance / 50 km/h + 0.5 hr buffer
            expected_sla = (distance_km / 50.0) + 0.5
            delayed = delivery_time_hours > expected_sla

            delivery_time = dispatch_time + datetime.timedelta(hours=delivery_time_hours)

            writer.writerow({
                "delivery_id": delivery_id,
                "order_date": str(order_date),
                "dispatch_time": dispatch_time.strftime("%Y-%m-%d %H:%M:%S"),
                "delivery_time": delivery_time.strftime("%Y-%m-%d %H:%M:%S"),
                "route": route_name,
                "origin": origin,
                "destination": dest,
                "carrier": carrier_name,
                "weather": weather_name,
                "parcel_weight_kg": parcel_weight_kg,
                "distance_km": distance_km,
                "traffic_level": traffic_name,
                "delivery_time_hours": delivery_time_hours,
                "delayed": delayed
            })

    print(f"Generated {num_rows} fixture records at: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_fixture()
