# Data-quality report

| Issue | Count | Decision |
| --- | --- | --- |
| Raw rows read | 24277 | Starting point of the pipeline. |
| Unparseable UTC timestamp | 69 | Y/M/D/H/MIN/SEC contained impossible components (e.g. month 49). Row dropped: the time of the observation cannot be established. |
| Timestamp outside 2023 | 2 | Two-digit year 0 (-> 2000) or corrupted date. Dropped for the same reason. |
| Duplicate (trip, timestamp) | 0 | Same trip recorded twice at the same second; first occurrence kept. |
| Timestamp inconsistent with recording order | 26 | id order is 100 % monotonic with the odometer, so it defines the true sequence. Rows stamped 16-20 h away from their neighbours (corrupted hour/minute) were dropped. |
| Physically impossible values | 0 | Speed outside 0-200 km/h, pedals outside 0-100 %, SOC outside 0-100 %, current outside -200..400 A. Dropped. |
| Missing cells in raw file | 0 | The file has no NaN cells; missing data is encoded with sentinel values instead (see next rows). |
| ALT == 0 (missing altitude) | 4 | Route is driven between 850 and 1350 m, so 0 means 'no GPS fix'. Flagged (alt_missing), not deleted. |
| VOL outside 250-420 V | 18049 | The pack-voltage channel is corrupted on 7 of 9 trips; rows kept but flagged (vol_valid). |
| GPS outside Brazil bounding box | 21 | Sign bit lost / zeroed fixes. Flagged (gps_valid), not deleted; GPS is not a model input. |
| Rows remaining after cleaning | 24180 | 0.40 % of raw rows removed. |

## Sampling gaps per trip

Gaps are reported, never interpolated.

| COND | n_rows | start_utc | end_utc | median_dt_s | n_gaps | largest_gap_s | total_gap_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2942 | 2023-10-23 17:17:10+00:00 | 2023-10-23 18:36:15+00:00 | 2.0 | 0 | 0.0 | 0.0 |
| 2 | 2653 | 2023-10-24 00:27:35+00:00 | 2023-10-24 01:39:15+00:00 | 2.0 | 0 | 0.0 | 0.0 |
| 3 | 2634 | 2023-10-24 11:55:57+00:00 | 2023-10-24 13:07:24+00:00 | 2.0 | 0 | 0.0 | 0.0 |
| 4 | 2992 | 2023-10-24 14:20:22+00:00 | 2023-10-24 15:40:16+00:00 | 2.0 | 0 | 0.0 | 0.0 |
| 5 | 2825 | 2023-10-24 17:22:44+00:00 | 2023-10-24 18:37:51+00:00 | 2.0 | 0 | 0.0 | 0.0 |
| 6 | 3211 | 2023-10-24 21:59:46+00:00 | 2023-10-24 23:25:40+00:00 | 2.0 | 0 | 0.0 | 0.0 |
| 7 | 3144 | 2023-10-26 16:56:09+00:00 | 2023-10-26 18:20:57+00:00 | 2.0 | 0 | 0.0 | 0.0 |
| 8 | 2987 | 2023-10-26 18:48:56+00:00 | 2023-10-26 20:14:12+00:00 | 2.0 | 1 | 322.0 | 322.0 |
| 9 | 792 | 2023-10-23 12:54:22+00:00 | 2023-10-23 14:30:44+00:00 | 2.0 | 1 | 4507.0 | 4507.0 |

## Trip summary after cleaning

| COND | rows | start_utc | end_utc | duration_min | median_dt_s | distance_km |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2942 | 2023-10-23 17:17:10+00:00 | 2023-10-23 18:36:15+00:00 | 79.1 | 2.0 | 76.4 |
| 2 | 2653 | 2023-10-24 00:27:35+00:00 | 2023-10-24 01:39:15+00:00 | 71.7 | 2.0 | 75.8 |
| 3 | 2634 | 2023-10-24 11:55:57+00:00 | 2023-10-24 13:07:24+00:00 | 71.5 | 2.0 | 76.3 |
| 4 | 2992 | 2023-10-24 14:20:22+00:00 | 2023-10-24 15:40:16+00:00 | 79.9 | 2.0 | 76.0 |
| 5 | 2825 | 2023-10-24 17:22:44+00:00 | 2023-10-24 18:37:51+00:00 | 75.1 | 2.0 | 68.9 |
| 6 | 3211 | 2023-10-24 21:59:46+00:00 | 2023-10-24 23:25:40+00:00 | 85.9 | 2.0 | 76.0 |
| 7 | 3144 | 2023-10-26 16:56:09+00:00 | 2023-10-26 18:20:57+00:00 | 84.8 | 2.0 | 76.2 |
| 8 | 2987 | 2023-10-26 18:48:56+00:00 | 2023-10-26 20:14:12+00:00 | 85.3 | 2.0 | 76.2 |
| 9 | 792 | 2023-10-23 12:54:22+00:00 | 2023-10-23 14:30:44+00:00 | 96.4 | 2.0 | 88.4 |
