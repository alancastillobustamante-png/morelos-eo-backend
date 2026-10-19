import os
import json
import time
from datetime import datetime, timezone

import ee
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

ROI_BBOX = [-99.30, 18.80, -98.90, 19.05]  # MVP corridor: Cuernavaca-Cuautla only
EE_PROJECT_ID = os.environ.get("EE_PROJECT_ID", "cleancuerna1")

# SAR flood layer only needs recomputing a few times a day — Sentinel-1
# revisits the same spot every 6-12 days, so anything faster is wasted work.
_flood_cache = {"data": None, "computed_at": 0}
FLOOD_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ee_key_json = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
        if ee_key_json:
            # Modo producción (Render) usando la Service Account
            key_dict = json.loads(ee_key_json)
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=ee_key_json
            )
            ee.Initialize(credentials, project=EE_PROJECT_ID)
        else:
            # Modo local: intenta usar el proyecto, y si falla usa la cuenta personal
            try:
                ee.Initialize(project=EE_PROJECT_ID)
            except Exception:
                ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"GEE Initialization Error: {e}")
    yield


app = FastAPI(
    title="Morelos EO Flood Risk Pipeline",
    description="Automated Sentinel-1 SAR & GPM Fusion Engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_roi_geometry():
    return ee.Geometry.BBox(*ROI_BBOX)


@app.get("/api/v1/health")
def health_check():
    return {"status": "online", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/v1/flood-extent")
def compute_sar_flood(
    days_back: int = 12,
    water_threshold_db: float = -18.0,
    before_start: Optional[str] = Query(None, description="YYYY-MM-DD, for historical validation"),
    after_start: Optional[str] = Query(None, description="YYYY-MM-DD, for historical validation"),
    force_refresh: bool = False,
):
    now_ts = time.time()
    cache_key_is_live = before_start is None and after_start is None

    if cache_key_is_live and not force_refresh and _flood_cache["data"] is not None:
        if now_ts - _flood_cache["computed_at"] < FLOOD_CACHE_TTL_SECONDS:
            return _flood_cache["data"]

    try:
        roi = get_roi_geometry()

        if before_start and after_start:
            # Historical mode — use this to validate against the May 2026 event
            after_date = ee.Date(after_start)
            before_date = ee.Date(before_start)
            after_end = after_date.advance(days_back, "day")
        else:
            # Live mode — always looking at "now"
            now = ee.Date(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            after_date = now.advance(-days_back, "day")
            after_end = now
            before_date = after_date.advance(-15, "day")

        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select("VH")
        )

        before = s1.filterDate(before_date, after_date).mean()
        after = s1.filterDate(after_date, after_end).mean()

        before_filtered = before.focal_median(50, "circle", "meters")
        after_filtered = after.focal_median(50, "circle", "meters")

        after_water = after_filtered.lt(water_threshold_db)
        before_water = before_filtered.lt(water_threshold_db)

        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem)
        flat_mask = slope.lt(5)

        after_clean = after_water.updateMask(flat_mask)
        before_clean = before_water.updateMask(flat_mask)

        new_flooding = after_clean.And(before_clean.Not()).selfMask()

        vectors = new_flooding.reduceToVectors(
            geometry=roi,
            crs="EPSG:4326",
            scale=30,
            geometryType="polygon",
            eightConnected=False,
            maxPixels=1e8,
        )

        geojson_data = vectors.getInfo()

        if cache_key_is_live:
            _flood_cache["data"] = geojson_data
            _flood_cache["computed_at"] = now_ts

        return geojson_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rainfall")
def compute_gpm_rainfall(hours_back: int = 48):
    try:
        roi = get_roi_geometry()
        now = ee.Date(datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        start = now.advance(-hours_back, "hour")

        gpm = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
            .filterBounds(roi)
            .filterDate(start, now)
            .select("precipitationCal")
        )

        cumulative_rain = gpm.sum().clip(roi)
        map_id = cumulative_rain.getMapId({"min": 0, "max": 100, "palette": ["white", "blue", "purple"]})

        return {"tile_url": map_id["tile_fetcher"].url_format, "period_hours": hours_back}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))